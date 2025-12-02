""" Single-step-multi-label, human clarification, SayCan environment

Combine data with LM response to the multiple choices, for conformal prediction.

"""
import os
import argparse
import pickle
import logging
import numpy as np
from omegaconf import OmegaConf
# use tokenizer from huggingface to get number of tokens of a string
from transformers import AutoTokenizer


def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Load answer data
    mc_post_data_path = os.path.join(
        cfg.parent_data_folder, cfg.mc_post_data_path_from_parent
    )
    with open(mc_post_data_path, 'rb') as f:
        data_all = pickle.load(f)

    # Load answer response data
    mc_post_response_data_path = os.path.join(
        cfg.parent_data_folder, cfg.mc_post_response_data_path_from_parent
    )
    # mc_post_response_data_path_2 = os.path.join(
    #     cfg.parent_data_folder, cfg.mc_post_response_data_path_from_parent_2
    # )
    if cfg.use_palm:
        mc_post_response_data_all = np.loadtxt(
            mc_post_response_data_path, dtype=str, delimiter='\t',
            encoding='utf-8-sig'
        )
        mc_post_response_data_all = mc_post_response_data_all.reshape(
            -1, 5
        )  #!
        # mc_post_response_data_all_2 = np.loadtxt(
        #     mc_post_response_data_path_2, dtype=str, delimiter='\t',
        #     encoding='utf-8-sig'
        # )
        # #
        # mc_post_response_data_all_2 = mc_post_response_data_all_2.reshape(
        #     -1, 5
        # )
        # mc_post_response_data_all = np.concatenate([
        #     mc_post_response_data_all, mc_post_response_data_all_2
        # ], axis=0)
    else:
        with open(mc_post_response_data_path, 'rb') as f:
            mc_post_response_data_all = pickle.load(f)

    # Generate data
    answer_data_all = []
    for data_ind, (data, mc_post_response_data) in enumerate(
        zip(data_all, mc_post_response_data_all)
    ):
        if data_ind < cfg.start_ind:
            continue

        # Extract
        if cfg.use_palm:
            # logprob_split = mc_post_response_data.split(' ')
            logprob_split = [float(v) for v in mc_post_response_data]
            
            # get perplexity
            if cfg.use_perplexity:
                mc_num_token_all = []
                perplexity_all = []
                for mc_ind, mc in enumerate(data['mc_post']['mc_all']):
                    num_token = len(tokenizer.encode(mc))
                    perplexity = np.exp(logprob_split[mc_ind] / num_token)
                    perplexity_all.append(perplexity)
                logprob_split = perplexity_all
                print(logprob_split)
            if len(logprob_split) < 5 or any(
                logprob == '' for logprob in logprob_split
            ):
                logging.info('Missing scores - rerun the data')
                top_logprobs = None
            else:
                top_logprobs = [float(v) for v in logprob_split]
            top_tokens = ['A', 'B', 'C', 'D', 'E']
        else:
            top_logprobs_full = mc_post_response_data['response_full'][
                "choices"][0]["logprobs"]["top_logprobs"][0]
            top_tokens = [token.strip() for token in top_logprobs_full.keys()]
            top_logprobs = [value for value in top_logprobs_full.values()]

        # logging.info(
        #     f"============== Probe {data_ind+1}/{cfg.num_data} =============="
        # )
        # logging.info(data['mc_prompt'])
        # logging.info("=======")

        # # Save
        data['top_tokens'] = top_tokens
        data['top_logprobs'] = top_logprobs

        # Save
        answer_data_all.append(data)

    # Save
    with open(cfg.data_save_path, 'wb') as f:
        pickle.dump(answer_data_all, f)

    # Summary
    logging.info('\n============== Summary ==============')
    logging.info(f'Number of questions generated: {cfg.num_data}')
    logging.info(f'Data saved to: {cfg.data_save_path}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)
    cfg.data_folder = os.path.dirname(args.cfg_file)
    cfg.parent_data_folder = os.path.dirname(cfg.data_folder)

    # Logging
    cfg.logging_path = os.path.join(
        cfg.data_folder, cfg.log_file_name + '.log'
    )
    logging.basicConfig(
        level=logging.INFO, format='%(message)s', handlers=[
            logging.FileHandler(cfg.logging_path, mode='w'),
            logging.StreamHandler()
        ]
    )

    # Save path
    cfg.data_save_path = os.path.join(
        cfg.data_folder, cfg.data_save_name + '.pkl'
    )
    cfg.single_data_save_dir = os.path.join(cfg.data_folder, 'single_data')

    # run
    main(cfg)
