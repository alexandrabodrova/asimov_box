""" Single-step, tabletop manipulation environment

Collect dataset of prompting LM to choose from the multiple choices (i.e., exeucting the action), also record true label from human.

"""
import os
import argparse
import pickle
import logging
from omegaconf import OmegaConf


def main(cfg):

    # Load background prompt
    with open(
        os.path.join(
            cfg.parent_data_folder, cfg.background_prompt_path_from_parent
        ), 'r'
    ) as f:
        background_prompt = f.read()

    # Load mc_post prompt
    with open(
        os.path.join(cfg.parent_data_folder, cfg.post_prompt_path_from_parent),
        'r'
    ) as f:
        raw_post_prompt = f.read()

    # Load previous data
    prev_data_path = os.path.join(
        cfg.parent_data_folder, cfg.prev_data_path_from_parent
    )
    with open(prev_data_path, 'rb') as f:
        data_all = pickle.load(f)

    # Load mc response data
    pre_response_data_path = os.path.join(
        cfg.parent_data_folder, cfg.pre_response_data_path_from_parent
    )
    if cfg.use_palm:
        with open(pre_response_data_path, 'r') as f:
            pre_response_data_all = f.read().split('--0000--')
    else:
        with open(pre_response_data_path, 'rb') as f:
            pre_response_data_all = pickle.load(f)

    # Generate data
    post_data_all = []
    post_prompt_all = []
    flag_prompt_verify = True
    for data_ind, (data, pre_response_data) in enumerate(
        zip(data_all, pre_response_data_all)
    ):

        if cfg.use_palm:
            pre_response = pre_response_data.strip()
        else:
            pre_response = pre_response_data['response'].strip()

        # process pre_response
        pre_response = pre_response.split('We:')[0].strip()

        # Get action prompt
        post_prompt = raw_post_prompt.replace(
            '{background}', background_prompt
        )
        post_prompt = post_prompt.replace(
            '{scene_description}', data['init']['scene_description']
        )
        post_prompt = post_prompt.replace('{request}', data['init']['request'])
        post_prompt = post_prompt.replace('{action}', pre_response)

        # Verify action prompt once
        if flag_prompt_verify:
            logging.info("===========")
            input(post_prompt + '\n\nPress any key to verify...')
            flag_prompt_verify = False

        # Save
        data['post'] = {}
        data['post']['background_prompt'] = background_prompt
        data['post']['post_prompt'] = post_prompt
        data['post']['pre_response'] = pre_response

        # Save
        post_data_all.append(data)
        post_prompt_all.append(post_prompt)
        logging.info("========================\n")

    # Save
    with open(cfg.data_save_path, 'wb') as f:
        pickle.dump(post_data_all, f)

    # Save text file
    with open(cfg.txt_save_path, 'w') as f:
        f.write('--0000--'.join(post_prompt_all))

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