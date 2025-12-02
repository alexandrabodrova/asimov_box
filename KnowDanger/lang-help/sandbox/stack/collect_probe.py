""" Multi-step-multi-label, human clarification, stacking, tabletop manipulation environment

Basic:
    - Collect dataset of prmopts for probing LM for clarification question. Save prompts as text file with delimiter '--0000--'.
    - Human help is triggered by LLM expressing itself being uncertain about the answer, instead of based on conformal prediction.

Context summarization:
    - Before probing for clarification at each step, we need to summarize the context including the previous steps, and the current scene (i.e., top object at the stack).

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

    # Load init state data
    init_data_path = os.path.join(
        cfg.parent_data_folder, cfg.init_data_path_from_parent
    )
    with open(init_data_path, 'rb') as f:
        init_data_all = pickle.load(f)

    # Load background prompt
    with open(
        os.path.join(cfg.parent_data_folder, cfg.background_prompt_probe_path),
        'r'
    ) as f:
        cfg.background_prompt = f.read()

    # Generate data
    probe_data_all = []
    probe_prompt_all = []
    for data in init_data_all:

        # Extract init data to get task prompt
        task_prompt = cfg.task_prompt.replace('{request}', data['request'])

        #! Get scene prompt - summarize previous steps
        scene_prompt = cfg.scene_prompt
        if cur_step >= 1:
            prev_action = ''
        if cur_step >= 2:  # add action from step 1
            prev_action += cfg.prev_action_prompt.replace(
                '{step}', '1'
            ).replace('{obj}', data['action'][0][0]
                     ).replace('{loc}', data['action'][0][1])
        if cur_step >= 3:  # add action from step 2
            prev_action += cfg.prev_action_prompt.replace(
                '{step}', '2'
            ).replace('{obj}', data['action'][1][0]
                     ).replace('{loc}', data['action'][1][1])
        scene_prompt = scene_prompt.replace('{prev_action}', prev_action)

        # Get the object at the top of the stack - assume previous step moved the object at the top every time
        scene_prompt = scene_prompt.replace(
            '{top_object}', data['stack'][cur_step - 1]
        )

        # Probe prompt
        probe_prompt = cfg.background_prompt + '\n\n' \
            + scene_prompt + '\n\n' \
            + task_prompt + '\n\n' \
            + cfg.probe_prompt

        # Add nested dict for current step
        data[f'step_{cur_step}'] = {}
        data[f'step_{cur_step}']['scene_prompt'] = scene_prompt
        data[f'step_{cur_step}']['task_prompt'] = task_prompt
        data[f'step_{cur_step}']['probe_prompt'] = probe_prompt

        # Save
        probe_data_all.append(data)
        probe_prompt_all.append(probe_prompt)

    # Save
    with open(cfg.data_save_path, 'wb') as f:
        pickle.dump(probe_data_all, f)

    # Save text file
    with open(cfg.txt_save_path, 'w') as f:
        f.write('--0000--'.join(probe_prompt_all))

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
