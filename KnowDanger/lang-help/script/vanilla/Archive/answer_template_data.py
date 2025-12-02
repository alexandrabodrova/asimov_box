""" Prompt language model to answer multiple choice question generated with templates.

No clarification round.

"""
import os
import logging
import argparse
from omegaconf import OmegaConf
import pickle
from tqdm import tqdm
import openai

from agent.language_model import LanguageModel


def main(cfg, log=True):
    # Language agent
    lm_agent = LanguageModel(cfg.openai_api_key)

    # Reload generated requests and multiple choices
    with open(cfg.request_data_path, 'rb') as handle:
        request_data = pickle.load(handle)

    # Load LM data if available, for debugging purposes and saving money - otherwise, save LM data in this run
    if cfg.lm_response_load_path != '':
        with open(cfg.lm_response_load_path, 'rb') as handle:
            lm_response_data = pickle.load(handle)
        logging.warning(
            f'Loaded LM response data: {cfg.lm_response_load_path}'
        )
        flag_load_lm_response_data = True
    else:
        lm_response_data = []
        flag_load_lm_response_data = False

    # Load prompt prefix - describing environment, or an example of request, multiple choices, and answer
    prompt_prefix = cfg.prompt_prefix

    # Determine single-step vs. multi-step
    num_step = cfg.num_step
    flag_multi_step = num_step > 1

    # Process each data
    num_data = len(request_data)
    data_type_split = [
        len([data for data in request_data if data['true_type'] == 'eq']),
        len([data for data in request_data if data['true_type'] == 'amb']),
        len([data for data in request_data if data['true_type'] == 'sem']),
    ]
    lm_answer_data = []
    for data_ind in tqdm(range(num_data)):
        data = request_data[data_ind]

        # For convenience, use nested list for single step too
        if flag_multi_step:
            context_steps = data['context_steps']
            true_label_steps = data['true_label_steps']
            true_type = data['true_type']
            mc_types_steps = data['mc_types_steps']
        else:
            context_steps = [data['context']]
            true_label_steps = data['true_label']
            true_type = data['true_type']
            mc_types_steps = [data['mc_types']]

        # run all steps
        lm_response_steps = []
        for step, context in enumerate(context_steps):

            # Append request and multiple choices
            full_prompt = prompt_prefix + context
            logging.warning(full_prompt) if log else None

            # Prompt LM
            if not flag_load_lm_response_data:
                response_full, response = lm_agent.prompt_gpt_complete(
                    full_prompt,
                    lm_model=cfg.lm_model,
                    temperature=cfg.lm_temperature,
                    stop=['#', 'objects ='],
                    max_tokens=cfg.lm_max_tokens,
                    logprobs=cfg.lm_logprobs,
                    logit_bias=cfg.include_token_id_bias,
                    timeout_seconds=cfg.lm_timeout_seconds,
                )
                lm_response_steps.append(response_full)
                logging.warning(response) if log else None
                logging.warning('') if log else None
            else:
                lm_response_saved = lm_response_data[data_ind][
                    step] if flag_multi_step else lm_response_data[data_ind]
                lm_response_steps.append(lm_response_saved)

        # Get logprobs and top tokens
        top_logprobs_full = response_full["choices"][0]["logprobs"][
            "top_logprobs"][0]
        top_tokens = [token.strip() for token in top_logprobs_full.keys()]
        top_logprobs = [value for value in top_logprobs_full.values()]

        # For convenience, remove inner list for single step
        context = context_steps[0] if not flag_multi_step else context_steps
        true_label = true_label_steps[
            0] if not flag_multi_step else true_label_steps
        lm_response = lm_response_steps[
            0] if not flag_multi_step else lm_response_steps
        mc_types = mc_types_steps[0] if not flag_multi_step else mc_types_steps

        # Save LM data
        lm_response_data.append(lm_response)

        # Save lm answer data
        lm_answer_data.append({
            'context': context,
            'full_prompt': full_prompt,
            'top_tokens': top_tokens,
            'top_logprobs': top_logprobs,
            'true_label': true_label,
            'true_type': true_type,
            'lm_response': lm_response,
            'mc_types': mc_types,
        })

    # Save data immediately to save money...
    if not flag_load_lm_response_data:
        with open(cfg.lm_response_save_path, 'wb') as handle:
            pickle.dump(
                lm_response_data, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    # Save request and LM data together for conformal prediction
    with open(cfg.lm_answer_save_path, 'wb') as handle:
        pickle.dump(lm_answer_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Report
    logging.warning('\n============== Summary ==============')
    logging.warning(f'Total number of data: {num_data}')
    logging.warning(f'True type split (eq, amb, sem): {data_type_split}')
    if flag_multi_step:
        true_label_rate = sum([
            x_steps["true_label"][step]
            == x_steps["lm_response"][step]["choices"][0]["text"].strip()
            for x_steps in lm_answer_data
            for step in range(num_step)
        ]) / (num_data*num_step)
    else:
        true_label_rate = sum([
            x["true_label"] == x["lm_response"]["choices"][0]["text"].strip()
            for x in lm_answer_data
        ]) / num_data
    logging.warning(
        f'Ratio of true label reported as the highest logprob by LM: {true_label_rate}'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    cfg.logging_path = os.path.join(
        cfg.answer_folder, cfg.lm_response_save_file_name + '.log'
    )
    cfg.request_data_path = os.path.join(
        cfg.collect_folder, cfg.request_load_file_name + '.pkl'
    )
    cfg.lm_response_save_path = os.path.join(
        cfg.answer_folder, cfg.lm_response_save_file_name + '.pkl'
    )
    cfg.lm_answer_save_path = os.path.join(
        cfg.answer_folder, cfg.lm_answer_save_file_name + '.pkl'
    )
    if cfg.lm_response_load_file_name != '':
        cfg.lm_response_load_path = os.path.join(
            cfg.answer_folder, cfg.lm_response_load_file_name + '.pkl'
        )
    else:
        cfg.lm_response_load_path = ''

    # logging
    logging.basicConfig(
        level=logging.INFO, format='%(message)s', handlers=[
            logging.FileHandler(cfg.logging_path, mode='w'),
            logging.StreamHandler()
        ]
    )  # overwrite
    openai.util.logging.getLogger().setLevel(
        logging.WARNING
    )  # suppress openai logging, but have to use warning for other logging

    # run
    main(cfg)
