""" Multi-step, human clarification, partially observable, tabletop manipulation environment

Collect the initial prompt.

URGENT:
1. save each part of prompt separately so it is easier to combine them for new prompts in the following rounds/steps.
2. save ground truth!!!

TODO:
1. Implement collecting the prompt for rest of the step (clarification, multiple choices)

"""

import os
import argparse
import pickle
import logging
import random
from omegaconf import OmegaConf

from agent.task import Task


def main(cfg):
    # Task agent
    task_agent = Task(cfg)

    # Generate data
    data_all = []
    for data_ind in range(cfg.num_data):

        # Sample request and the ground truth
        request, info = task_agent.sample_request()

        # Collect the prompt
        task_prompt = cfg.task_prompt.replace('{request}', request)
        prompt = cfg.background_prompt + '\n\n' + cfg.scene_prompt + '\n\n' + task_prompt + '\n\n' + cfg.probe_uncertainty_prompt

        # Log
        logging.info(
            '================= Data {} ================='.
            format(len(data_all) + 1)
        )
        logging.info(
            f'Request: {request} - ambiguity type: {info.ambiguity_type}'
        )
        logging.info(f'Ground truth: {info.request_unambiguous}')
        logging.info(prompt)
        logging.info('================= END =================\n\n')

        # Save data
        data_all.append({
            'prompt': prompt,
            'info': info,
        })

    # Save all data
    with open(cfg.data_save_path, 'wb') as f:
        pickle.dump(data_all, f)

    # Summary
    logging.info('\n============== Summary ==============')
    logging.info(f'Number of questions generated: {len(data_all)}')
    logging.info(f'Data saved to: {cfg.data_save_path}.')
    logging.info('=====================================')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    cfg.logging_path = os.path.join(
        cfg.data_folder, cfg.log_file_name + '.log'
    )
    cfg.data_save_path = os.path.join(
        cfg.data_folder, cfg.data_save_name + '.pkl'
    )

    # logging
    logging.basicConfig(
        level=logging.INFO, format='%(message)s', handlers=[
            logging.FileHandler(cfg.logging_path, mode='w'),
            logging.StreamHandler()
        ]
    )

    # run
    random.seed(cfg.seed)
    main(cfg)
