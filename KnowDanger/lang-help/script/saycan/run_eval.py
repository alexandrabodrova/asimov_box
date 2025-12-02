"""
Obtain the final actions to be executed with SayCan robot.

Perform calibration, and save plan success.

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


def check_help_success(cfg, prediction_set, data):
    if 'add_mc_prefix' in data:
        none_option_token = data['add_mc_prefix']
    else:
        none_option_token = cfg.e_sig  # default to 'e/E'

    # ask for help if prediction set is not a singleton or 'e/E' is in the prediction set
    flag_help = False
    if cfg.count_e_as_help:
        cond = len(prediction_set) != 1 or none_option_token in prediction_set
    else:
        cond = len(prediction_set) != 1
    if cond:
        flag_help = True

    # check success
    if flag_help:
        if cfg.help_mode == 'from_prediction_set':
            if cfg.multi_label:
                flag_success = not set(true_label).isdisjoint(prediction_set)
            else:
                flag_success = true_label in prediction_set
        elif cfg.help_mode == 'from_all_mc':  # always work
            flag_success = True
        else:
            raise 'Unknown help model!'
    else:
        assert len(prediction_set) == 1
        flag_success = not set(true_label).isdisjoint(prediction_set)
    return flag_help, flag_success


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

    # save path
    save_path = os.path.join(cfg.save_dir, 'test.pkl')

    # conformal
    agent = ConformalPredictor(cfg)
    agent.calibrate(log=False)

    # to be saved
    final_data = []

    # stats
    cp_empirical_coverage = 0
    naive_empirical_coverage = 0
    cp_help = 0
    cp_success = 0
    naive_help = 0
    naive_success = 0
    num_diff_action = 0
    for data in agent.test_data:

        # Get information
        true_label = data['true_label']
        top_tokens = data['top_tokens']
        top_logprobs = data['top_logprobs']

        # temperature scaling and get softmax
        top_smx = temperature_scaling(top_logprobs, cfg.temperature_scaling)
        data['top_smx'] = top_smx

        # Get prediction set
        cp_prediction_set = get_prediction_set(
            top_tokens, top_smx, agent.qhat, 'conformal', cfg
        )
        naive_prediction_set = get_prediction_set(
            top_tokens, top_smx, cfg.naive_cal_level, 'naive', cfg
        )
        data['cp_prediction_set'] = cp_prediction_set
        data['naive_prediction_set'] = naive_prediction_set

        # Check help and success
        cp_flag_help, cp_flag_success = check_help_success(
            cfg, cp_prediction_set, data
        )
        naive_flag_help, naive_flag_success = check_help_success(
            cfg, naive_prediction_set, data
        )
        data['cp_flag_help'] = cp_flag_help
        data['cp_flag_success'] = cp_flag_success
        data['naive_flag_help'] = naive_flag_help
        data['naive_flag_success'] = naive_flag_success
        cp_help += cp_flag_help
        cp_success += cp_flag_success
        naive_help += naive_flag_help
        naive_success += naive_flag_success

        # find action to be taken by CP and naive - mark if different
        true_label = data['true_label']
        if cp_flag_help:
            try:
                cp_action = random.choice(
                    list(set(true_label).intersection(cp_prediction_set))
                )
            except:
                cp_action = random.choice(true_label)
        else:
            assert len(cp_prediction_set) == 1
            cp_action = random.choice(cp_prediction_set[0])
        if naive_flag_help:
            try:
                naive_action = random.choice(
                    list(set(true_label).intersection(naive_prediction_set))
                )
            except:
                naive_action = random.choice(true_label)
        else:
            assert len(naive_prediction_set) == 1
            naive_action = random.choice(naive_prediction_set[0])

        # Check if action different
        data['flag_diff'] = cp_action != naive_action
        if data['flag_diff']:
            num_diff_action += 1

        # actual action
        conversion = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        cp_action_actual = data['mc_all'][conversion[cp_action]]
        naive_action_actual = data['mc_all'][conversion[naive_action]]
        data['cp_action_actual'] = cp_action_actual
        data['naive_action_actual'] = naive_action_actual

        # set action to human specified one if LLM chooses None of the others
        if data['cp_action_actual'] == 'a different option not listed here':
            data['cp_action_actual'] = data['true_action']
            assert data['true_action'] is not None
        if data['naive_action_actual'] == 'a different option not listed here':
            data['naive_action_actual'] = data['true_action']
            assert data['true_action'] is not None

        # save until target number of real trials reached
        if len(final_data) < cfg.num_trial:
            final_data.append(data)

    logging.info('CP help: {:.4f}'.format(cp_help / len(agent.test_data)))
    logging.info(
        'CP success: {:.4f}'.format(cp_success / len(agent.test_data))
    )
    logging.info(
        'Naive help: {:.4f}'.format(naive_help / len(agent.test_data))
    )
    logging.info(
        'Naive success: {:.4f}'.format(naive_success / len(agent.test_data))
    )
    logging.info(
        'Number of different actions: {:.4f}'.format(
            num_diff_action / len(agent.test_data)
        )
    )

    # Save data
    with open(save_path, 'wb') as f:
        pickle.dump(final_data, f)
