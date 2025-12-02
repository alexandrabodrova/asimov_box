""" Multi-step-multi-label, stacking, tabletop manipulation environment

Collect prompts for LM to geenerate possible choices.

"""
import os
import argparse
import pickle
import logging
from omegaconf import OmegaConf


def main(cfg):
    # Current step
    cur_step = cfg.current_step

    # Load background and mc_pre prompt
    with open(
        os.path.join(
            cfg.parent_data_folder,
            cfg.background_prompt_mc_pre_path_from_parent
        ), 'r'
    ) as f:
        background_prompt = f.read()
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
        prev_data_all = pickle.load(f)

    # Generate data
    mc_pre_data_all = []
    mc_pre_prompt_all = []
    flag_no_clarify_verified = False
    for data_ind, data in enumerate(prev_data_all):

        # Print probe prompt
        logging.info(
            f"============== Probe {data_ind+1}/{cfg.num_data} =============="
        )
        logging.info(f"Task: {data['init']['request']}")

        # Get scene prompt - summarize previous steps
        if cur_step >= 1:
            prev_action = ''
        if cur_step >= 2:  # add action from step 1
            prev_action += cfg.prev_action_prompt.replace(
                '{step}', 'first'
            ).replace('{obj}', data['action'][0][0]
                     ).replace('{loc}', data['action'][0][1])
        if cur_step >= 3:  # add action from step 2
            prev_action += ' ' + cfg.prev_action_prompt.replace(
                '{step}', 'second'
            ).replace('{obj}', data['action'][1][0]
                     ).replace('{loc}', data['action'][1][1])

        # get mc prompt
        mc_pre_prompt = raw_mc_pre_prompt.replace(
            '{background}', background_prompt
        )
        mc_pre_prompt = mc_pre_prompt.replace(
            '{scene_description}', data['init']['scene_description']
        )
        mc_pre_prompt = mc_pre_prompt.replace(
            '{request}', data['init']['request']
        )
        mc_pre_prompt = mc_pre_prompt.replace(
            '{thought}', data['init']['thought']
        )
        if cur_step > 1:
            mc_pre_prompt = mc_pre_prompt.replace('{prev_action}', prev_action)

        # Verify answer prompt
        if not flag_no_clarify_verified:
            logging.info("============== Answer ==============")
            logging.info(mc_pre_prompt)
            logging.info("=======\n")
            input("Press any key to confirm the answer prompt.\n")
            flag_no_clarify_verified = True

        # Save
        data[f'step_{cur_step}'] = {}
        data[f'step_{cur_step}']['mc_pre_prompt'] = mc_pre_prompt
        data[f'step_{cur_step}']['prev_action'] = prev_action

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
    cfg.parent_data_folder = os.path.dirname(os.path.dirname(cfg.data_folder))

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