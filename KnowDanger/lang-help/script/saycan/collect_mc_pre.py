""" Single-step-multi-label, human clarification, Saycan environment

Collect dataset of human providing claridications and prompts for LM to geenerate possible choices.

"""

import os
import argparse
import pickle
import logging
from omegaconf import OmegaConf


def main(cfg):
    # Load prompts
    with open(cfg.background_prompt_path, 'r') as f:
        raw_background_prompt = f.read()
    with open(cfg.mc_pre_prompt_path, 'r') as f:
        raw_mc_pre_prompt = f.read()

    # Load probe data
    probe_data_path = os.path.join(
        cfg.parent_data_folder, cfg.probe_data_path_from_parent
    )
    with open(probe_data_path, 'rb') as f:
        data_all = pickle.load(f)

    # Load LM response data
    probe_response_data_path = os.path.join(
        cfg.parent_data_folder, cfg.probe_response_data_path_from_parent
    )
    # with open(probe_response_data_path, 'rb') as f:
    #     probe_response_data_all = pickle.load(f)
    with open(probe_response_data_path, 'r') as f:
        probe_response_data_all = f.read().split('--0000--')
    assert cfg.num_data == len(probe_response_data_all)
    assert cfg.num_data == len(data_all)

    # Generate data
    mc_pre_data_all = []
    mc_pre_prompt_all = []
    flag_clarify_verified = False
    flag_no_clarify_verified = False
    for data_ind, (data, probe_response_data) in enumerate(
        zip(data_all, probe_response_data_all)
    ):
        # Remove trailing spaces and newlines
        probe_lm_response = probe_response_data.strip()

        # Print probe prompt
        logging.info(
            f"============== Probe {data_ind+1}/{cfg.num_data} =============="
        )
        logging.info(f"\n(Task: {data['probe_prompt']})")
        logging.info(f"(Ground truth: {data['true_labels']})")
        logging.info('LM probe response: {}'.format(probe_lm_response))
        logging.info("=======")

        # Check if clarification is needed
        flag_clarify = ('uncertain' in probe_lm_response.lower())

        # New prompt to be appended to background
        mc_pre_prompt = raw_mc_pre_prompt.replace(
            '{task}', data['task_prompt']
        )
        mc_pre_prompt = mc_pre_prompt.replace(
            '{obj_list_on_counter}', data['obj_scene_description']
        )
        mc_pre_prompt = mc_pre_prompt.replace(
            '{special_scene_description}', data['special_scene_description']
        )
        mc_pre_prompt = mc_pre_prompt.replace(
            '{probe_response}', probe_lm_response
        )

        # Prompt human for clarification
        if flag_clarify:
            if cfg.dummy_clarify:  # for test purposes
                human_response = "Sorry I can't answer right now."
            else:
                while 1:
                    try:
                        human_response = input('Please help LM:\n')
                    except:
                        continue
                    break

            # Add human response and action to prompt
            action_prompt = cfg.action_prompt
            mc_pre_prompt = mc_pre_prompt.replace(
                '{human_feedback}', '\nWe: ' + human_response
            )
            mc_pre_prompt = mc_pre_prompt.replace(
                '{mc_pre_prompt}', action_prompt
            )
            mc_pre_prompt = raw_background_prompt + '\n\n' + mc_pre_prompt

            # Verify answer prompt
            if not flag_clarify_verified:
                logging.info("============== Answer ==============")
                logging.info(mc_pre_prompt)
                logging.info("=======\n")
                input("Press any key to confirm the answer prompt.\n")
                flag_clarify_verified = True
        else:
            input('No clarification needed. Press any key to continue.')
            human_response = ''

            # Add action to prompt
            action_prompt = cfg.action_no_clarify_prompt
            mc_pre_prompt = mc_pre_prompt.replace(
                '{human_feedback}', human_response
            )
            mc_pre_prompt = mc_pre_prompt.replace(
                '{mc_pre_prompt}', action_prompt
            )
            mc_pre_prompt = raw_background_prompt + '\n\n' + mc_pre_prompt

            # Verify answer prompt
            if not flag_no_clarify_verified:
                logging.info("============== Answer ==============")
                logging.info(mc_pre_prompt)
                logging.info("=======\n")
                input("Press any key to confirm the answer prompt.\n")
                flag_no_clarify_verified = True

        # Save
        data['mc_pre_prompt'] = mc_pre_prompt
        data['probe_lm_response'] = probe_lm_response
        data['human_response'] = human_response
        data['action_prompt'] = action_prompt

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