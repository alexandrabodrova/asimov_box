""" Single-step-multi-label, human clarification, SayCan environment

Collect dataset of prompting LM to score each multiple choice with perplexity.

"""

import os
import argparse
import pickle
import logging
import random
from omegaconf import OmegaConf


def main(cfg):
    # Load prompts
    background_prompt_path = os.path.join(
        cfg.parent_data_folder, cfg.background_prompt_path
    )
    with open(background_prompt_path, 'r') as f:
        raw_background_prompt = f.read()
    mc_post_prompt_path = os.path.join(
        cfg.parent_data_folder, cfg.mc_post_prompt_path
    )
    with open(mc_post_prompt_path, 'r') as f:
        raw_mc_post_prompt = f.read()

    # Load previous data
    prev_data_path = os.path.join(
        cfg.parent_data_folder, cfg.prev_data_path_from_parent
    )
    with open(prev_data_path, 'rb') as f:
        prev_data_all = pickle.load(f)

    # Generate data
    mc_post_prompt_all = []
    mc_answer_all = []
    flag_prompt_verify = True
    for data_ind, data in enumerate(prev_data_all):

        mc_all = data['mc_all']
        mc_sigs = ['A) ', 'B) ', 'C) ', 'D) ', 'E) ']
        for mc_ind, mc in enumerate(mc_all):

            # New prompt to be appended to background
            try:
                mc_post_prompt = raw_mc_post_prompt.replace(
                    '{task}', data['request']
                )
            except:
                mc_post_prompt = raw_mc_post_prompt.replace(
                    '{task}', data['prompt']
                )
            mc_post_prompt = mc_post_prompt.replace(
                '{obj_list_on_counter}', data['obj_scene_description']
            )
            mc_post_prompt = mc_post_prompt.replace(
                '{special_scene_description}',
                data['special_scene_description']
            )
            mc_post_prompt = mc_post_prompt.replace(
                '{multiple_choices}', data['mc_prompt']
            )
            mc_post_prompt = raw_background_prompt + '\n\n' + mc_post_prompt

            # Verify action prompt once
            if flag_prompt_verify:
                input(
                    '\n\n' + mc_post_prompt + '\n\nPress any key to verify...'
                )
                flag_prompt_verify = False

            # Add sig for answer
            # mc_answer = mc_sigs[mc_ind] + mc
            mc_answer = mc

            # Save
            mc_post_prompt_all.append(mc_post_prompt)
            mc_answer_all.append(mc_answer)
        logging.info("=======\n")

    # Save text file
    with open(cfg.txt_save_path, 'w') as f:
        f.write('--0000--'.join(mc_post_prompt_all))
    with open(cfg.mc_save_path, 'w') as f:
        f.write('--0000--'.join(mc_answer_all))

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
    cfg.mc_save_path = os.path.join(
        cfg.data_folder, cfg.data_save_name + '_mc.txt'
    )

    # run
    random.seed(cfg.seed)
    main(cfg)
