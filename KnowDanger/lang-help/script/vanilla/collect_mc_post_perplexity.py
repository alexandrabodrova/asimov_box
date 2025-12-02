""" 
Collect dataset of prompting LM to score each multiple choice with perplexity.

"""

import os
import argparse
import pickle
import logging
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

        mc_all = data['mc_post']['mc_all']
        for mc_ind, mc in enumerate(mc_all):

            # New prompt to be appended to background
            mc_post_prompt = raw_mc_post_prompt.replace(
                '{request}', data['init']['request'].lower()
            )
            mc_post_prompt = mc_post_prompt.replace(
                '{scene_description}', data['init']['scene_description']
            )
            mc_post_prompt = mc_post_prompt.replace(
                '{background}', raw_background_prompt
            )

            # Verify action prompt once
            if flag_prompt_verify:
                print(mc)
                input(
                    '\n\n' + mc_post_prompt + '\n\nPress any key to verify...'
                )
                flag_prompt_verify = False

            # Save
            mc_post_prompt_all.append(mc_post_prompt)
            mc_answer_all.append(mc)
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
    main(cfg)