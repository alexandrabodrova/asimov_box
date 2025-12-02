"""
Check cases where CP succeeds but Naive fails, or vice versa.

Only the single-step template setting now.

"""
import argparse
import os
from omegaconf import OmegaConf
import logging
import numpy as np

from agent.predict.conformal_predictor import ConformalPredictor
from agent.predict.util import get_score, get_prediction_set, temperature_scaling


def main(args, cfg):
    np.set_printoptions(precision=3)

    # Calibrate first
    agent = ConformalPredictor(cfg)
    agent.calibrate()

    # Loop over test data
    num_help = 0
    num_success = 0
    num_correct_prediction_set = 0
    for data in agent.test_data:
        # top_logprobs_full = data['lm_response']["choices"][0]["logprobs"][
        # "top_logprobs"][0]
        # top_tokens = [token.strip() for token in top_logprobs_full.keys()]
        # top_logprobs = [value for value in top_logprobs_full.values()]
        true_label = data['true_label']
        top_logprobs = data['top_logprobs']
        top_tokens = data['top_tokens']
        none_option_token = data['add_mc_prefix']

        # temperature scaling and get softmax
        top_smx = temperature_scaling(top_logprobs, cfg.temperature_scaling)

        # Get prediction set with CP
        prediction_set_cp = get_prediction_set(
            top_tokens, top_smx, agent.qhat, cfg.score_method, cfg
        )

        # Get prediction set with naive
        # cfg.naive_cal_level = 0.90
        prediction_set_naive = get_prediction_set(
            top_tokens, top_smx, agent.qhat, 'naive', cfg
        )

        # print(prediction_set_cp, prediction_set_naive)

        def get_success_help(prediction_set):
            # ask for help if prediction set is not a singleton or 'e/E' is in the prediction set
            flag_help = False
            if cfg.count_e_as_help:
                cond = len(
                    prediction_set
                ) != 1 or none_option_token in prediction_set
            else:
                cond = len(prediction_set) != 1
            if cond:
                flag_help = True

            # check success
            flag_success = False
            if flag_help:
                if cfg.help_mode == 'from_prediction_set':
                    flag_success = not set(true_label
                                          ).isdisjoint(prediction_set)
                elif cfg.help_mode == 'from_all_mc':  # always work
                    flag_success = True
                else:
                    raise 'Unknown help model!'
            else:
                assert len(prediction_set) == 1
                flag_success = not set(true_label).isdisjoint(prediction_set)
            return flag_success, flag_help

        # Get success and help flags
        flag_success_cp, flag_help_cp = get_success_help(prediction_set_cp)
        flag_success_naive, flag_help_naive = get_success_help(
            prediction_set_naive
        )

        # Log if specified
        if flag_success_cp and flag_help_naive and not flag_help_cp:
            # if flag_success_naive and not flag_success_cp:
            logging.info('----------------------------------------')
            try:
                logging.info(data['request'])
            except:
                logging.info(data['task_prompt'])
            logging.info(data['scene_objects'])
            logging.info(data['mc_post_prompt'])
            logging.info(
                f'True label: {true_label}; Prediction set with CP: {prediction_set_cp}; Prediction set with Naive: {prediction_set_naive}'
            )
            logging.info(
                f'Prob: {[np.round(np.exp(top_logprobs[top_tokens.index(sig)]), 3) for sig in cfg.mc_sigs if sig in top_tokens]}'
            )
            logging.info(
                f'Scaled softmax: {[np.round(top_smx[top_tokens.index(sig)], 3) for sig in cfg.mc_sigs if sig in top_tokens]}'
            )
            logging.info('----------------------------------------\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default='/home/allen/lang-help/data/saycan/collect',
        help="Base data directory",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Temperature for temperature scaling",
    )
    args = parser.parse_args()

    cfg = OmegaConf.create()
    cfg.seed = 42
    cfg.base_dir = args.base_dir
    cfg.load_data_path = os.path.join(cfg.base_dir, 'answer/answer.pkl')
    # cfg.save_dir = os.path.join(cfg.base_dir, 'calibrate')
    cfg.calibration_ratio = 0.8
    cfg.count_e_as_help = False
    cfg.help_mode = 'from_prediction_set'  # from_all_mc, from_prediction_set
    # cfg.temperature_scaling = 3
    cfg.temperature_scaling = args.temperature
    cfg.alpha = 0.25
    cfg.score_method = 'conformal'
    cfg.naive_cal_level = 0.58
    cfg.logging_path = os.path.join(cfg.base_dir, 'evaluate_failure_inv.log')
    cfg.mc_sigs = ['A', 'B', 'C', 'D', 'E']
    # cfg.e_sig = 'E'
    cfg.multi_label = True

    # logging
    if cfg.logging_path is not None:
        logging.basicConfig(
            level=logging.INFO, format='%(message)s', handlers=[
                logging.FileHandler(cfg.logging_path, mode='w'),
                logging.StreamHandler()
            ]
        )  # overwrite

    main(args, cfg)
