""" Multi-step-multi-label, human clarification, stacking, tabletop manipulation environment

Collect dataset of human providing claridications and prompts for LM to geenerate possible choices.

"""
import os
import argparse
import pickle
import logging
import random
from omegaconf import OmegaConf


def main(cfg):
    # Current step
    cur_step = cfg.current_step

    # Load probe data
    probe_data_path = os.path.join(
        cfg.parent_data_folder, cfg.probe_data_path_from_parent
    )
    with open(probe_data_path, 'rb') as f:
        data_all = pickle.load(f)

    # Load probe response data
    probe_response_data_path = os.path.join(
        cfg.parent_data_folder, cfg.probe_response_data_path_from_parent
    )
    with open(probe_response_data_path, 'rb') as f:
        probe_response_data_all = pickle.load(f)

    # Load background prompt
    with open(
        os.path.join(cfg.parent_data_folder, cfg.background_prompt_mc_path),
        'r'
    ) as f:
        cfg.background_prompt = f.read()

    # Generate data
    mc_pre_data_all = []
    mc_pre_prompt_all = []
    flag_clarify_verified = False
    flag_no_clarify_verified = False
    for data_ind, (data, probe_response_data) in enumerate(
        zip(data_all, probe_response_data_all)
    ):
        probe_response = probe_response_data['response']

        # Print probe prompt
        logging.info(
            f"============== Probe {data_ind+1}/{cfg.num_data} =============="
        )
        # logging.info(data_ind['probe_prompt'])
        logging.info(probe_response)
        # help human respond
        logging.info(f"\n(Task: {data['request']})")
        # logging.info(f"(Ground truth: {data['info']['request_unambiguous']})")
        logging.info("=======")

        # Check if clarification is needed
        flag_clarify = ('uncertain' in probe_response.lower())

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

            # new prompt - need to replace the old background prompt with one for mc
            mc_pre_prompt = cfg.background_prompt + '\n\n' \
                + data[f'step_{cur_step}']['scene_prompt'] + '\n\n' \
                + data[f'step_{cur_step}']['task_prompt'] + '\n\n' \
                + cfg.probe_prompt + ' ' + probe_response + '\n\n' \
                + 'We: ' + human_response + '\n\n' \
                + cfg.action_prompt

            # Verify answer prompt
            if not flag_clarify_verified:
                logging.info("============== Answer ==============")
                logging.info(mc_pre_prompt)
                logging.info("=======\n")
                input("Press any key to confirm the answer prompt.\n")
                flag_clarify_verified = True
        else:
            input('No clarification needed. Press any key to continue.')
            mc_pre_prompt = data['probe_prompt'] + ' ' + probe_response + '\n\n' \
                + cfg.action_no_clarify_prompt

            # Verify answer prompt
            if not flag_no_clarify_verified:
                logging.info("============== Answer ==============")
                logging.info(mc_pre_prompt)
                logging.info("=======\n")
                input("Press any key to confirm the answer prompt.\n")
                flag_no_clarify_verified = True

        # Save
        data[f'step_{cur_step}']['mc_pre_prompt'] = mc_pre_prompt
        data[f'step_{cur_step}']['probe_response'] = probe_response

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
