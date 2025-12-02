""" Single-step-multi-label, human clarification, SayCan environment

Collect dataset of prompting LM to choose from the multiple choices (i.e., exeucting the action).

For re-using data from v1 and v3 combined.

"""

import os
import argparse
import pickle
import logging
import random
from omegaconf import OmegaConf
from agent.multiple_choice import MultipleChoice


def main(cfg):

    # Load previous data
    prev_data_path = os.path.join(
        cfg.parent_data_folder, cfg.prev_data_path_from_parent
    )
    with open(prev_data_path, 'rb') as f:
        prev_data_all = pickle.load(f)

    # Generate data
    mc_post_prompt_all = []
    flag_prompt_verify = True
    for data_ind, data in enumerate(prev_data_all):

        mc_post_prompt = data['mc_post_prompt']

        # Verify action prompt once
        if flag_prompt_verify:
            input('\n\n' + mc_post_prompt + '\n\nPress any key to verify...')
            flag_prompt_verify = False

        # Save prompt
        mc_post_prompt_all.append(mc_post_prompt)
        logging.info("=======\n")

    # Save
    # with open(cfg.data_save_path, 'wb') as f:
    #     pickle.dump(mc_post_data_all, f)

    # Save text file
    with open(cfg.txt_save_path, 'w') as f:
        f.write('--0000--'.join(mc_post_prompt_all))

    # Summary
    logging.info('\n============== Summary ==============')
    # logging.info(f'Number of questions generated: {cfg.num_data}')
    # logging.info(f'Data saved to: {cfg.data_save_path}.')
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

    # run
    random.seed(cfg.seed)
    main(cfg)
