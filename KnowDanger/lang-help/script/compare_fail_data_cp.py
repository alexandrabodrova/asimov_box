"""
Check the failure examples of different baselines.

Case:
- Naive includes additional option while CP give a singleton of true label
- Naive misses the true label, but CP includes it with a larger prediction set

Since Naive is not calibrated, we need to first find the calibration level so that Naive and CP have the same empirical coverage.

"""
import argparse
import numpy as np
import logging
from omegaconf import OmegaConf
import seaborn as sns
from agent.predict.conformal_predictor import ConformalPredictor
from agent.predict.util import get_score, get_prediction_set, temperature_scaling


sns.set(font_scale=2.5)
sns.set_style('darkgrid', {'axes.linewidth': 3, 'axes.edgecolor': 'black'})


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


def log_info(data):
    logging.info('Prompt: {}'.format(data['mc_post_prompt']))
    logging.info('True label: {}'.format(data['true_label']))
    logging.info('Top tokens: {}'.format(data['top_tokens']))
    logging.info('Top logprobs: {}'.format(data['top_logprobs']))
    logging.info('CP prediction set: {}'.format(data['cp_prediction_set']))
    logging.info(
        'Naive prediction set: {}'.format(data['naive_prediction_set'])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)

    # logging
    logging.basicConfig(
        level=logging.INFO, format='%(message)s', handlers=[
            logging.FileHandler(cfg.logging_path, mode='w'),
            logging.StreamHandler()
        ]
    )

    # temperature scaling
    prev_tc = 0.5
    cfg.score_method = 'conformal'
    cfg.temperature_scaling = prev_tc
    agent = ConformalPredictor(cfg)
    agent.calibrate(log=False)
    prev_qhat = agent.qhat

    # conformal
    tc = 1
    cfg.score_method = 'conformal'
    cfg.temperature_scaling = tc
    agent = ConformalPredictor(cfg)
    agent.calibrate(log=False)

    # manually go through the test set and find the failure examples
    data_all_cp_succeed_naive_fail = []
    data_all_cp_fail_naive_succeed = []
    data_all_cp_avoid_help = []
    data_all_naive_avoid_help = []
    cp_empirical_coverage = 0
    naive_empirical_coverage = 0

    cp_help = 0
    cp_success = 0
    naive_help = 0
    naive_success = 0
    for data in agent.test_data:

        # Get information
        # data['prompt'] = data['context'
        #                      ]  # TODO (repo-wise): do not use context as key
        true_label = data['true_label']
        top_tokens = data['top_tokens']
        top_logprobs = data['top_logprobs']

        # temperature scaling and get softmax
        top_smx = temperature_scaling(top_logprobs, cfg.temperature_scaling)
        data['top_smx'] = top_smx
        data['tc'] = tc

        #
        top_smx_prev = temperature_scaling(top_logprobs, prev_tc)

        # Get prediction set
        cp_prediction_set = get_prediction_set(
            top_tokens, top_smx, agent.qhat, 'conformal', cfg
        )
        naive_prediction_set = get_prediction_set(
            top_tokens, top_smx_prev, prev_qhat, 'conformal', cfg
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

        # Check when CP succeeds but Naive fails - this is when Naive misses the true label, but CP includes it with a larger prediction set
        if cp_flag_success and not naive_flag_success:
            data_all_cp_succeed_naive_fail.append(data)

        # Check when CP fails but Naive succeeds
        if not cp_flag_success and naive_flag_success:
            data_all_cp_fail_naive_succeed.append(data)

        # Check when Naive includes additional option while CP give a singleton of true label
        if len(naive_prediction_set) > len(cp_prediction_set) and not set(
            true_label
        ).isdisjoint(cp_prediction_set) and not cp_flag_help:
            assert naive_flag_help
            data_all_cp_avoid_help.append(data)

        # Check when Naive avoids help
        if len(naive_prediction_set) < len(cp_prediction_set) and not set(
            true_label
        ).isdisjoint(naive_prediction_set) and not naive_flag_help:
            assert cp_flag_help
            data_all_naive_avoid_help.append(data)

        # Check CP empirical coverage - make sure it is the same as Naive
        if not set(true_label).isdisjoint(cp_prediction_set):
            cp_empirical_coverage += 1 / len(agent.test_data)
        if not set(true_label).isdisjoint(naive_prediction_set):
            naive_empirical_coverage += 1 / len(agent.test_data)
    logging.info('CP empirical coverage: {}'.format(cp_empirical_coverage))
    logging.info(
        'Naive empirical coverage: {}'.format(naive_empirical_coverage)
    )

    # Print info
    logging.info(
        'Checking cases where CP succeeds but Naive fails - this is when Naive misses the true label, but CP includes it with a larger prediction set'
    )
    logging.info('-------------------------------------------------')
    for ind, data in enumerate(data_all_cp_succeed_naive_fail):
        logging.info('Example {}'.format(ind))
        log_info(data)
        logging.info('-------------------------------------------------')
    logging.info('=================================================\n\n')

    logging.info('Checking cases where CP fails but Naive succeeds')
    logging.info('-------------------------------------------------')
    for ind, data in enumerate(data_all_cp_fail_naive_succeed):
        logging.info('Example {}'.format(ind))
        log_info(data)
        logging.info('-------------------------------------------------')
    logging.info('=================================================\n\n')

    logging.info(
        'Checking cases where Naive includes additional option while CP give a singleton of true label'
    )
    logging.info('-------------------------------------------------')
    for ind, data in enumerate(data_all_cp_avoid_help):
        logging.info('Example {}'.format(ind))
        log_info(data)
        logging.info('-------------------------------------------------')
    logging.info('=================================================\n\n')

    logging.info('Checking cases where Naive avoids help')
    logging.info('-------------------------------------------------')
    for ind, data in enumerate(data_all_naive_avoid_help):
        logging.info('Example {}'.format(ind))
        log_info(data)
        logging.info('-------------------------------------------------')
    logging.info('=================================================\n\n')

    logging.info('CP help: {}'.format(cp_help))
    logging.info('CP success: {}'.format(cp_success))
    logging.info('Naive help: {}'.format(naive_help))
    logging.info('Naive success: {}'.format(naive_success))
    logging.info('CP success/help: {}'.format(cp_success / cp_help))
    logging.info('Naive success/help: {}'.format(naive_success / naive_help))