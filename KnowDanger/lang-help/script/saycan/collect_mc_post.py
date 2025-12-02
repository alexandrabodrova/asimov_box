""" Single-step-multi-label, human clarification, SayCan environment

Collect dataset of prompting LM to choose from the multiple choices (i.e., exeucting the action).

"""

import os
import argparse
import pickle
import logging
import random
from omegaconf import OmegaConf
from agent.multiple_choice import MultipleChoice


def main(cfg):
    # Load prompts
    with open(cfg.background_prompt_path, 'r') as f:
        raw_background_prompt = f.read()
    with open(cfg.mc_post_prompt_path, 'r') as f:
        raw_mc_post_prompt = f.read()

    # Multiple choice agent
    mc_agent = MultipleChoice()

    # Load answer data
    mc_pre_data_path = os.path.join(
        cfg.parent_data_folder, cfg.mc_pre_data_path_from_parent
    )
    with open(mc_pre_data_path, 'rb') as f:
        data_all = pickle.load(f)

    # Load answer response data
    mc_pre_response_data_path = os.path.join(
        cfg.parent_data_folder, cfg.mc_pre_response_data_path_from_parent
    )
    # with open(mc_pre_response_data_path, 'rb') as f:
    #     mc_pre_response_data_all = pickle.load(f)
    with open(mc_pre_response_data_path, 'r') as f:
        mc_pre_response_data_all = f.read().split('--0000--')

    # Generate data
    mc_post_data_all = []
    mc_post_prompt_all = []
    flag_prompt_verify = True
    for data_ind, (data, mc_pre_response_data) in enumerate(
        zip(data_all, mc_pre_response_data_all)
    ):
        # Process multiple choice - randomize order
        # mc_pre_response = mc_pre_response_data['response']
        mc_pre_response = mc_pre_response_data.strip()
        mc_prompt, mc_all, add_mc_prefix = mc_agent.process_multiple_choice(
            mc_pre_response, add_mc=cfg.add_mc
        )

        # Correct human response, yikes...
        if 'uncertain' not in data['probe_lm_response'].lower():
            human_response = ''
        else:
            human_response = '\nWe: ' + data['human_response']

        # New prompt to be appended to background
        mc_post_prompt = raw_mc_post_prompt.replace(
            '{task}', data['task_prompt']
        )
        mc_post_prompt = mc_post_prompt.replace(
            '{obj_list_on_counter}', data['obj_scene_description']
        )
        mc_post_prompt = mc_post_prompt.replace(
            '{special_scene_description}', data['special_scene_description']
        )
        mc_post_prompt = mc_post_prompt.replace(
            '{probe_response}', data['probe_lm_response']
        )
        mc_post_prompt = mc_post_prompt.replace(
            '{human_feedback}', human_response
        )
        mc_post_prompt = mc_post_prompt.replace(
            '{action_prompt}', data['action_prompt']
        )
        mc_post_prompt = mc_post_prompt.replace(
            '{multiple_choices}', mc_prompt
        )
        mc_post_prompt = mc_post_prompt.replace(
            '{answer_prompt}', cfg.answer_prompt
        )
        mc_post_prompt = raw_background_prompt + '\n\n' + mc_post_prompt

        # Verify action prompt once
        if flag_prompt_verify:
            input('\n\n' + mc_post_prompt + '\n\nPress any key to verify...')
            flag_prompt_verify = False

        # Save
        data['mc_post_prompt'] = mc_post_prompt
        data['mc_all'] = mc_all
        data['mc_prompt'] = mc_prompt
        data['mc_pre_response'] = mc_pre_response
        data['add_mc_prefix'] = add_mc_prefix

        # Save
        mc_post_data_all.append(data)
        mc_post_prompt_all.append(mc_post_prompt)
        logging.info("=======\n")

    # Save
    with open(cfg.data_save_path, 'wb') as f:
        pickle.dump(mc_post_data_all, f)

    # Save text file
    with open(cfg.txt_save_path, 'w') as f:
        f.write('--0000--'.join(mc_post_prompt_all))

    # Summary
    logging.info('\n============== Summary ==============')
    logging.info(f'Number of questions generated: {cfg.num_data}')
    logging.info(f'Data saved to: {cfg.data_save_path}.')
    logging.info(f'Prompt saved to: {cfg.txt_save_path}.')


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

    # Merge with prompt cfg
    cfg_prompt = OmegaConf.load(
        os.path.join(cfg.parent_data_folder, 'prompt.yaml')
    )
    cfg = OmegaConf.merge(cfg_prompt, cfg)

    # Merge with common cfg
    cfg_common = OmegaConf.load(
        os.path.join(cfg.parent_data_folder, 'common.yaml')
    )
    cfg = OmegaConf.merge(cfg_common, cfg)

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
    cfg.txt_save_path = os.path.join(
        cfg.data_folder, cfg.data_save_name + '.txt'
    )

    # run
    random.seed(cfg.seed)
    main(cfg)
