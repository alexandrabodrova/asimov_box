"""
Get final actions and scenes to run on hardware. Rename actions if needed.

"""
import argparse
import os
import numpy as np
import pickle
import logging
import random
from omegaconf import OmegaConf
from agent.predict.conformal_predictor import ConformalPredictor
from agent.predict.util import get_score, get_prediction_set, temperature_scaling

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    random.seed(cfg.seed)

    # logging
    logging.basicConfig(
        level=logging.INFO, format='%(message)s', handlers=[
            logging.FileHandler(cfg.logging_path, mode='w'),
            logging.StreamHandler()
        ]
    )

    # load data
    with open(cfg.load_data_path, 'rb') as f:
        data_all = pickle.load(f)

    # save path
    save_path = os.path.join(cfg.save_dir, 'action.pkl')

    #
    final_data = []
    extra_action = []
    for data_ind, data in enumerate(data_all):

        # Get information
        scene_objects = data['scene_objects']
        cp_action = data['cp_action_actual']
        naive_action = data['naive_action_actual']

        # prompt human for renaming action
        print(f"{data_ind}: CP action is {cp_action}")
        # new_action = input("Please input new action, or press enter to skip")
        # if new_action != '':
        #     cp_action = new_action
        #     print('Action renamed to', cp_action)

        # print cp action
        logging.info(f'{data_ind}: {scene_objects}, {cp_action}')
        try:
            logging.info(f"{data['request']}")
        except:
            print(data.keys())
            logging.info(f"{data['task_prompt']}")
        logging.info(f"{data['mc_prompt']}")

        if data['flag_diff']:
            print(f"Naive action is {naive_action}")
            # new_action = input(
            #     "Please input new action, or press enter to skip"
            # )
            # if new_action != '':
            #     naive_action = new_action
            #     print('Action renamed to', naive_action)
            extra_action.append([data_ind, scene_objects, naive_action])

        # save data
        data['cp_action_corrected'] = cp_action
        data['naive_action_corrected'] = naive_action
        final_data.append(data)
        print('====')

    # print naive action
    for data_ind, scene_objects, naive_action in extra_action:
        logging.info(f'{data_ind}: {scene_objects}, {naive_action}')
        logging.info(f"{data['mc_prompt']}")

    # Save data
    with open(save_path, 'wb') as f:
        pickle.dump(final_data, f)
