""" Single-step-single-label, Bimanual environment

"""

import os
import argparse
import pickle
import logging
from omegaconf import OmegaConf


def main(cfg):

    # Load prompts
    with open(
        os.path.join(
            cfg.parent_data_folder,
            cfg.background_prompt_mc_pre_path_from_parent
        ), 'r'
    ) as f:
        raw_background_prompt = f.read()
    with open(
        os.path.join(
            cfg.parent_data_folder, cfg.mc_pre_prompt_path_from_parent
        ), 'r'
    ) as f:
        raw_mc_pre_prompt = f.read()

    # Load prev data
    prev_data_path = os.path.join(
        cfg.parent_data_folder, cfg.prev_data_path_from_parent
    )
    with open(prev_data_path, 'rb') as f:
        data_all = pickle.load(f)

    # Generate data
    mc_pre_data_all = []
    mc_pre_prompt_all = []
    flag_verified = False
    for data_ind, data in enumerate(data_all):

        # Print probe prompt
        logging.info(f"======= {data_ind+1}/{cfg.num_data} =======")
        logging.info(f"(Task: {data['request']})")
        logging.info("=====================")

        # Open example txt
        example_txt_path = os.path.join(cfg.data_folder, data['example_file'])
        with open(example_txt_path, 'r') as f:
            example_prompt = f.read()

        # New prompt to be appended to background
        mc_pre_prompt = raw_mc_pre_prompt.replace('{request}', data['request'])
        mc_pre_prompt = mc_pre_prompt.replace(
            '{additional_background}', data['additional_background']
        )
        mc_pre_prompt = mc_pre_prompt.replace(
            '{background}', raw_background_prompt
        )
        mc_pre_prompt = mc_pre_prompt.replace('{example}', example_prompt)

        # Verify answer prompt
        if not flag_verified:
            logging.info("============== Answer ==============")
            logging.info(mc_pre_prompt)
            logging.info("=======\n")
            input("Press any key to confirm the answer prompt.\n")
            flag_verified = True

        # Save
        data['mc_pre_prompt'] = mc_pre_prompt

        # Save
        mc_pre_data_all.append(data)
        mc_pre_prompt_all.append(mc_pre_prompt)
        logging.info("============================================ \n")

    # Save
    with open(cfg.data_save_path, 'wb') as f:
        pickle.dump(mc_pre_data_all, f)

    # Save text file
    with open(cfg.txt_save_path, 'w') as f:
        f.write('--0000--'.join(mc_pre_prompt_all))

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

    # run
    main(cfg)