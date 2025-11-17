""" Single-step, human clarification, tabletop manipulation environment

Collect dataset of LLM-Human interactions with LLM generating multiple choice predictions, asking clarifications, and human answering. Human help is triggered by LLM expressing itself being uncertain about the answer, instead of based on conformal prediction.

Collect full trials at once --- not recommended given language model API timeout. Instead, use collect_prompt.py to collect the initial prompt, and then ...

This file is for single-action setting with multiple rounds of clarifications. For now, do not use infeasible request (e.g., "move a plate to ..." but there is no plate on the table).

Workflow:
    For each data point:
        1. Prompt LLM to ask question if it is uncertain about the answer
        2. Human answers the question
        3. Repeat 1-2 until max number of clarification rounds is reached
        4. Prompt LLM to generate multiple choices
        5. Prompt LLM to choose between the multiple choice

We follow the ChatGPT convention to append new messages to the prompt:
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
where system is some background information, user is the human input, and assistant is the LM output. At prompt time, we use a custom function to convert the meesages to a single string.

TODOs:
1. Implement multi-action-step setting
2. Test GPT-4 once granted API access
3. Add custom requests (ones that are difficult to parameterize)

"""

import os
import argparse
import pickle
import logging
import random
from omegaconf import OmegaConf
import openai

from agent.task import Task
from agent.language_model import LanguageModel, convert_messages_to_prompt
from agent.multiple_choice import MultipleChoice
from agent.predict.util import get_prediction_set, temperature_scaling
from util.data import process_answer_response


def main(cfg):
    # Task agent
    task_agent = Task(cfg)

    # LM agent
    lm_agent = LanguageModel(cfg.openai_api_key)

    # Multiple choice agent
    mc_agent = MultipleChoice()

    # Verify system prompt
    logging.warning('\n----------------------------------------')
    logging.warning('System prompt:')
    logging.warning(cfg.background_prompt)
    logging.warning('----------------------------------------\n')

    # Generate data
    calibrate_data_all = []
    response_data_all = []
    data_ind = 1 + cfg.single_save_data_offset
    while data_ind <= cfg.num_data:

        ######################################################################
        ############################   Request   #############################
        ######################################################################

        # Sample request and the ground truth
        request, info = task_agent.sample_request()
        logging.warning(f'---------- Collecting data # {data_ind} ----------')
        logging.warning(
            f'Request: {request} - ambiguity type: {info.ambiguity_type}'
        )
        logging.warning(f'Ground truth: {info.request_unambiguous}')

        # Fill in system message (background) and request
        messages = [
            {
                "role": "system",
                "content": cfg.background_prompt
            },
            {
                "role": "user",
                "content": cfg.task_prompt.replace('{request}', request)
            },
        ]

        ######################################################################
        ######################   Clarification rounds   ######################
        ######################################################################

        clarify_response_all = []
        human_response_all = []
        mc_clarify_response_all = []
        for clarify_round in range(cfg.max_num_clarify_round):

            ####################################################################
            ###############  Determine if clarify; if not, break   #############
            ####################################################################

            # Trigger clarification based on prediction set (CP or LLM output)
            if cfg.clarify_mode == 'cp' or cfg.clarify_mode == 'prompt_set':

                # Get the multiple choices
                messages_clarify = messages.copy()
                messages_clarify.append({
                    "role": "user",
                    "content": cfg.mc_for_clarify_prompt + '\n\nRobot: ',
                })
                mc_clarify_response_full, mc_clarify_response = lm_agent.prompt_gpt_complete(
                    convert_messages_to_prompt(messages_clarify),
                    lm_model=cfg.lm_model_mc,
                    temperature=cfg.lm_temperature_mc,
                    max_tokens=cfg.lm_max_token_mc,
                    timeout_seconds=cfg.lm_timeout_seconds,
                )
                mc_clarify_response_all.append(mc_clarify_response_full)
                mc_clarify_all, success = mc_agent.process_multiple_choice(
                    mc_clarify_response
                )
                if cfg.clarify_mode == 'prompt_set':
                    mc_clarify_all += '\ne) None of the above'
                messages_clarify.append({
                    "role": "assistant",
                    "content": mc_clarify_all,
                })
                messages_clarify.append({
                    "role": "user",
                    "content": cfg.answer_for_clarify_prompt + '\n\nRobot: ',
                })

                # Answer, and then get prediction set with CP
                if cfg.clarify_mode == 'cp':
                    response_full, _ = lm_agent.prompt_gpt_complete(
                        convert_messages_to_prompt(messages_clarify),
                        lm_model=cfg.lm_model_answer,
                        temperature=0,
                        logit_bias=cfg.include_token_id_bias,
                        max_tokens=1,  # limit to a,b,c,d,e
                        logprobs=5,
                        timeout_seconds=cfg.lm_timeout_seconds,
                    )  # data not saved
                    top_logprobs_full = response_full["choices"][0][
                        "logprobs"]["top_logprobs"][0]
                    top_tokens = [
                        token.strip() for token in top_logprobs_full.keys()
                    ]
                    top_logprobs = [
                        value for value in top_logprobs_full.values()
                    ]
                    top_smx = temperature_scaling(
                        top_logprobs,
                        cfg.temperature_scaling,
                    )
                    prediction_set = get_prediction_set(
                        top_tokens,
                        top_smx,
                        cfg.qhat,
                        cfg.score_method,
                    )

                # or LLM outputting prediction set directly
                elif cfg.clarify_mode == 'prompt_set':
                    _, predict_response = lm_agent.prompt_gpt_complete(
                        convert_messages_to_prompt(messages_clarify),
                        lm_model=cfg.lm_model_set,
                        temperature=cfg.lm_temperature_set,
                        max_tokens=cfg.lm_max_token_set,
                        timeout_seconds=cfg.lm_timeout_seconds,
                    )
                    prediction_set = predict_response.strip().lower(
                    ).split(',')
                    for ind, prediction in enumerate(prediction_set):
                        prediction_set[ind] = prediction.split(')')[0]
                logging.warning(f'\nPrediction set: {prediction_set}')

                # If prediction set is singleton and not contains 'e/E', do not ask for clarification and terminate all rounds
                if len(
                    prediction_set
                ) == 1 and cfg.e_sig not in prediction_set:
                    flag_clarify = False
                    break

            # Trigger clarification with LLM reporting binary uncertainty
            elif cfg.clarify_mode == 'prompt_binary':
                messages_calibrate = messages.copy()
                messages_calibrate.append({
                    "role": "user",
                    "content": cfg.binary_prompt,
                })
                # logging.warning(convert_messages_to_prompt(messages_calibrate))
                _, binary_response = lm_agent.prompt_gpt_complete(
                    convert_messages_to_prompt(messages_calibrate),
                    lm_model=cfg.lm_model_ask,
                    temperature=cfg.lm_temperature_ask,
                    max_tokens=cfg.lm_max_token_ask,
                    timeout_seconds=cfg.lm_timeout_seconds,
                )
                logging.warning('Binary response:')
                logging.warning(binary_response)

                # do not clarify if report 'uncertain'
                if 'uncertain' not in binary_response.strip().lower():
                    flag_clarify = False
                    break
            else:
                raise 'Unknown clarify mode!'

            ####################################################################
            ##################   Ask clarification question   ##################
            ####################################################################

            flag_clarify = True
            messages.append({
                "role": "user",
                "content": cfg.clarify_prompt,
            })
            # logging.warning(convert_messages_to_prompt(messages))
            _, clarify_response = lm_agent.prompt_gpt_complete(
                convert_messages_to_prompt(messages) + 'Robot: ',
                lm_model=cfg.lm_model_clarify,
                temperature=cfg.lm_temperature_clarify,
                max_tokens=cfg.lm_max_token_clarify,
                timeout_seconds=cfg.lm_timeout_seconds,
            )
            clarify_response_all.append(clarify_response)
            logging.warning('LM response:')
            logging.warning(clarify_response)
            logging.warning('---------')

            # Add clarification response to messages
            messages.append({
                "role": "assistant",
                "content": 'Robot: ' + clarify_response,
            })

            ####################################################################
            #####################  Prompt human to clarify   ###################
            ####################################################################

            while 1:
                try:
                    human_response = input('Please help LM:\n')
                except:
                    continue
                break
            messages.append({
                "role": "user",
                "content": 'Human: ' + human_response,
            })
            human_response_all.append(human_response)

        ####################################################################
        ################   Generate multiple choices for action   ##########
        ####################################################################

        logging.warning('Clarification rounds done, now taking action')

        # Regardless of the calibration mode, always generate multiple choices for the action. For prompt_binary, during test time, we will prompt LLM again to see if it is certain about the action. Similarly for prompt_set, during test time, we will prompt LLM again to generate prediction set.

        # Genetate multiple choices for action
        if flag_clarify:
            mc_for_action_prompt = cfg.mc_for_action_prompt
        else:
            mc_for_action_prompt = cfg.mc_for_action_no_clarify_prompt
        messages.append({
            "role": "user",
            "content": mc_for_action_prompt,
        })
        # logging.warning(convert_messages_to_prompt(messages))
        mc_action_response_full, mc_action_response = lm_agent.prompt_gpt_complete(
            convert_messages_to_prompt(messages),
            lm_model=cfg.lm_model_mc,
            temperature=cfg.lm_temperature_mc,
            max_tokens=cfg.lm_max_token_mc,
            timeout_seconds=cfg.lm_timeout_seconds,
        )
        logging.warning('MC generated (raw):')
        logging.warning(mc_action_response)
        logging.warning('---------')
        mc_action_all, success = mc_agent.process_multiple_choice(
            mc_action_response
        )
        if not success:
            logging.warning(
                'Failed to process multiple choice, re-collect this data'
            )
            continue
        logging.warning('MC generated (processed):')
        logging.warning(mc_action_all)
        logging.warning('---------')
        messages.append({
            "role": "assistant",
            "content": mc_action_all,
        })

        ####################################################################
        #######################   Prompt human to label   ##################
        ####################################################################

        while 1:
            try:
                logging.warning(f'Again, the task is {request}')
                logging.warning(
                    f'And the ground truth is {info.request_unambiguous}'
                )
                true_label = input(
                    "Please provide label(s) in the format of 'label_1, label_2, ...'; e for none of the above: "
                ).split(',')
                if len(true_label) < 1:
                    raise ValueError
            except:
                continue
            break

        ####################################################################
        #########################   LM answers MC   ########################
        ####################################################################

        messages.append({
            "role": "user",
            "content": cfg.answer_for_action_prompt,
        })
        # logging.warning(convert_messages_to_prompt(messages))
        answer_action_response_full, answer_action_response = lm_agent.prompt_gpt_complete(
            convert_messages_to_prompt(messages),
            lm_model=cfg.lm_model_answer,
            temperature=0,
            logit_bias=cfg.include_token_id_bias,
            max_tokens=1,  # limit to a,b,c,d,e
            logprobs=5,
            timeout_seconds=cfg.lm_timeout_seconds,
        )
        top_logprobs_full, top_token, top_tokens, top_logprobs = process_answer_response(
            answer_action_response_full
        )
        logging.warning(f'\nLM answer: {answer_action_response}')
        logging.warning(f'Top tokens: {top_tokens}')
        logging.warning(f'logprobs: {top_logprobs}')
        logging.warning(f'True label(s): {true_label}')
        logging.warning('---------\n')

        ####################################################################
        ###########################   Save data   ##########################
        ####################################################################

        calibrate_data_all.append({
            'request': request,
            'messages': messages,
            'answer_response_full': answer_action_response_full,
            'mc': mc_action_all,
            'true_label': true_label,
        })
        response_data_all.append({
            'mc_clarify_response_all': mc_clarify_response_all,
            'clarify_response': clarify_response_all,
            'mc_action_response': mc_action_response_full,
            'answer_action_response': answer_action_response_full,
            'human_response_all': human_response_all,
        })

        # Save single data
        with open(
            os.path.join(
                cfg.single_data_save_folder, f'{data_ind}_calibrate.pkl'
            ), 'wb'
        ) as handle:
            pickle.dump(
                calibrate_data_all[-1], handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )
        with open(
            os.path.join(
                cfg.single_data_save_folder, f'{data_ind}_response.pkl'
            ), 'wb'
        ) as handle:
            pickle.dump(
                response_data_all[-1], handle, protocol=pickle.HIGHEST_PROTOCOL
            )

        # Quit if user input 'q', re-collect if 'a'
        data_instruction = input(
            'Press q to stop data collection, press a to re-collect this data, any other key to continue:\n'
        )
        if data_instruction == 'q':
            break
        elif data_instruction == 'a':
            continue

        # Count
        data_ind += 1

    # Summary
    logging.warning('\n============== Summary ==============')
    logging.warning(
        f'Number of questions generated: {len(calibrate_data_all)}'
    )
    logging.warning(
        f'Ratio of human label reported as the highest logprob by LM: {sum([x["answer_response_full"]["choices"][0]["text"].strip() in x["true_label"] for x in calibrate_data_all])/cfg.num_data}'
    )
    # logging.warning(f'Data saved to: {cfg.data_save_path}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    cfg.logging_path = os.path.join(
        cfg.data_folder, cfg.log_file_name + '.log'
    )
    cfg.data_save_path = os.path.join(
        cfg.data_folder, cfg.data_save_name + '.pkl'
    )
    cfg.single_data_save_folder = os.path.join(cfg.data_folder, 'single_data')
    if not os.path.exists(cfg.single_data_save_folder):
        os.makedirs(cfg.single_data_save_folder)

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
    random.seed(cfg.seed)
    main(cfg)
