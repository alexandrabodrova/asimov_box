""" Multi-step-multi-label, human clarification, stacking, tabletop manipulation environment

Collect dataset of prompting LM to choose from the multiple choices (i.e., exeucting the action), also record true label from human.

If human chooses E, then also need to label the true action.

We could also generate the true label automatiaclly, but not implemented yet.

"""
import os
import argparse
import pickle
import logging
import random
from omegaconf import OmegaConf
from agent.multiple_choice import MultipleChoice


def main(cfg):
    # Current step
    cur_step = cfg.current_step

    # Multiple choice agent
    mc_agent = MultipleChoice()

    # Load answer data
    mc_pre_data_path = os.path.join(
        cfg.parent_data_folder, cfg.mc_pre_data_path_from_parent
    )
    with open(mc_pre_data_path, 'rb') as f:
        data_all = pickle.load(f)

    # Load answer response data
    mc_pre_response_data_path = os.path.join(
        cfg.parent_data_folder, cfg.mc_pre_response_data_path_from_parent
    )
    with open(mc_pre_response_data_path, 'rb') as f:
        mc_pre_response_data_all = pickle.load(f)

    # Generate data
    mc_post_data_all = []
    mc_post_prompt_all = []
    flag_prompt_verify = True
    for data_ind, (data, mc_pre_response_data) in enumerate(
        zip(data_all, mc_pre_response_data_all)
    ):
        mc_pre_response = mc_pre_response_data['response']

        # Print probe prompt
        logging.info(
            f"============== Probe {data_ind+1}/{cfg.num_data} =============="
        )
        logging.info(data[f'step_{cur_step}']['mc_pre_prompt'])
        logging.info(mc_pre_response)
        logging.info("=======")

        # Process multiple choice
        mc_prompt, mc_all, success = mc_agent.process_multiple_choice(
            mc_pre_response
        )

        # TODO: show randomized order of choices!!!

        # Prompt human to label
        if cfg.dummy_label:
            true_label = [random.choice(['A', 'B', 'C', 'D'])]
        else:
            while 1:
                try:
                    logging.info(f"Again, the task is {data['request']}")
                    # logging.info(
                    #     f"And the ground truth is {data['request_unambiguous']}"
                    # )
                    true_label = input(
                        "Please provide label(s) in the format of 'label_1, label_2, ...'; E for none of the above: "
                    ).split(',')
                    if len(true_label) < 1:
                        raise ValueError
                except:
                    continue
                break

        # Prompt human to label true action if E is chosen - this should only happen with E as the only label
        true_action = None
        if 'E' in true_label:
            assert len(true_label) == 1
            while 1:
                try:
                    true_action = input(
                        "Please label the true action in the format of {object}_{action}"
                    )
                except:
                    continue
                break

        # Get action prompt
        mc_post_prompt = data[f'step_{cur_step}']['mc_pre_prompt'] + '\n' \
            + mc_prompt + '\n\n' \
            + cfg.mc_post_prompt

        # Verify action prompt once
        if flag_prompt_verify:
            input(mc_post_prompt + '\n\nPress any key to verify...')
            flag_prompt_verify = False

        # Save
        data[f'step_{cur_step}']['mc_all'] = mc_all
        data[f'step_{cur_step}']['mc_post_prompt'] = mc_post_prompt
        data[f'step_{cur_step}']['mc_pre_response'] = mc_pre_response
        data[f'step_{cur_step}']['true_label'] = true_label
        data[f'step_{cur_step}']['true_action'] = true_action

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
