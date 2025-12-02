""" Single-step-single-label, Bimanual environment

"""
import os
import argparse
import pickle
import logging
from omegaconf import OmegaConf
from agent.multiple_choice import MultipleChoice


def main(cfg):

    # Multiple choice agent
    mc_agent = MultipleChoice()

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
        raw_mc_post_prompt = f.read()

    # Load previous data
    prev_data_path = os.path.join(
        cfg.parent_data_folder, cfg.prev_data_path_from_parent
    )
    with open(prev_data_path, 'rb') as f:
        data_all = pickle.load(f)

    # Load mc response data if specified
    mc_pre_response_data_path = os.path.join(
        cfg.parent_data_folder, cfg.mc_pre_response_data_path_from_parent
    )
    with open(mc_pre_response_data_path, 'r') as f:
        mc_pre_response_data_all = f.read().split('--0000--')

    # Generate data
    mc_post_data_all = []
    mc_post_prompt_all = []
    flag_prompt_verify = True
    for data_ind, (data, mc_pre_response_data) in enumerate(
        zip(data_all, mc_pre_response_data_all)
    ):
        if cfg.use_palm:
            mc_pre_response = mc_pre_response_data.strip()
        else:
            mc_pre_response = mc_pre_response_data['response']

        # Print probe prompt
        logging.info(
            f"============== Probe {data_ind+1}/{cfg.num_data} =============="
        )
        # logging.info(data['mc_pre_prompt'])
        # logging.info(mc_pre_response)

        # Process multiple choice
        try:
            mc_prompt, mc_all, add_mc_prefix = mc_agent.process_multiple_choice(
                mc_pre_response, add_mc=cfg.add_mc
            )
        except:
            mc_all.append('do nothing')
            mc_prompt += '\nD) do nothing'

        # Get action prompt
        mc_post_prompt = raw_mc_post_prompt.replace(
            '{additional_background}', data['additional_background']
        )
        mc_post_prompt = mc_post_prompt.replace(
            '{background}', background_prompt
        )
        mc_post_prompt = mc_post_prompt.replace('{request}', data['request'])
        mc_post_prompt = mc_post_prompt.replace('{mc}', mc_prompt).strip()

        # Verify action prompt once
        if flag_prompt_verify:
            logging.info("===========")
            input(mc_post_prompt + '\n\nPress any key to verify...')
            flag_prompt_verify = False

        # Save
        data['mc_all'] = mc_all
        data['mc_prompt'] = mc_prompt
        data['mc_post_prompt'] = mc_post_prompt
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