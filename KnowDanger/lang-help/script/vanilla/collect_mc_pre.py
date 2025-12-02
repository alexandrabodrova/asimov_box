""" Single-step, tabletop manipulation environment

Collect dataset of prompts for LM to geenerate possible choices.

"""
import os
import argparse
import pickle
import logging
from omegaconf import OmegaConf


def main(cfg):
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

    # Verify system prompt
    logging.info('=========== Background prompt ==============')
    logging.info(background_prompt)
    logging.info('============================================\n')
    input('Press any key to verify...')

    # Load prev data
    prev_data_path = os.path.join(
        cfg.parent_data_folder, cfg.prev_data_path_from_parent
    )
    with open(prev_data_path, 'rb') as f:
        prev_data_all = pickle.load(f)

    # Load probe response data if specified
    if cfg.probe_response_data_path_from_parent:
        probe_response_data_path = os.path.join(
            cfg.parent_data_folder, cfg.probe_response_data_path
        )
        with open(probe_response_data_path, 'rb') as f:
            probe_response_data_all = pickle.load(f)
        flag_probe = True
    else:
        probe_response_data_all = [None] * cfg.num_data
        flag_probe = False

    # Generate data
    mc_pre_data_all = []
    mc_pre_prompt_all = []
    flag_clarify_verified = False
    flag_no_clarify_verified = False
    for data_ind, (data, probe_response_data) in enumerate(
        zip(prev_data_all, probe_response_data_all)
    ):
        if flag_probe:
            probe_response = probe_response_data['response']
            flag_clarify = ('uncertain' in probe_response.lower())
        else:
            probe_response = None
            flag_clarify = False

        # Print probe prompt
        logging.info(
            f"============== Probe {data_ind+1}/{cfg.num_data} =============="
        )
        logging.info(probe_response)
        logging.info(f"\n(Task: {data['init']['request']})")
        logging.info(f"(Ground truth: {data['init']['request_unambiguous']})")
        logging.info("=======")

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

            # Get mc prompt TODO: fill in other parts
            mc_pre_prompt = raw_mc_pre_prompt.replace(
                '{background}', background_prompt
            )
            mc_pre_prompt = mc_pre_prompt.replace(
                '{scene_description}', data['init']['scene_description']
            )
            mc_pre_prompt = mc_pre_prompt.replace(
                '{request}', data['init']['request']
            )

            # Verify answer prompt
            if not flag_clarify_verified:
                logging.info("============== Answer ==============")
                logging.info(mc_pre_prompt)
                logging.info("=======\n")
                input("Press any key to confirm the answer prompt.\n")
                flag_clarify_verified = True
        else:
            # input('No clarification needed. Press any key to continue.')

            # Get mc prompt
            mc_pre_prompt = raw_mc_pre_prompt.replace(
                '{background}', background_prompt
            )
            mc_pre_prompt = mc_pre_prompt.replace(
                '{scene_description}', data['init']['scene_description']
            )
            mc_pre_prompt = mc_pre_prompt.replace(
                '{request}', data['init']['request'].lower()
            )

            # Verify answer prompt
            if not flag_no_clarify_verified:
                logging.info("============== Answer ==============")
                logging.info(mc_pre_prompt)
                logging.info("=======\n")
                input("Press any key to confirm the answer prompt.\n")
                flag_no_clarify_verified = True

        # Save
        data['mc_pre'] = {}
        data['mc_pre']['background_prompt'] = background_prompt
        data['mc_pre']['mc_pre_prompt'] = mc_pre_prompt
        data['mc_pre']['probe_response'] = probe_response

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