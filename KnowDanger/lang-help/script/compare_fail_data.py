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


def check_help_success(cfg, prediction_set):
    # ask for help if prediction set is not a singleton or 'e/E' is in the prediction set
    flag_help = False
    if cfg.count_e_as_help:
        cond = len(prediction_set) != 1 or cfg.e_sig in prediction_set
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
        flag_success = true_label in prediction_set
    return flag_help, flag_success


def log_info(data):
    logging.info('Prompt: {}'.format(data['prompt']))
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
    tc_all = cfg.temperature_scaling_all
    assert len(tc_all) == 1
    tc = tc_all[0]

    # First, find the calibration level so that Naive and CP have the same empirical coverage.
    target_coverage = 1 - cfg.alpha + 0.013  # alpha is used by CP
    naive_cal_level_all = np.arange(0.5, 0.99, 0.03)
    naive_cal_level_all = np.hstack(
        (naive_cal_level_all, np.arange(0.991, 0.999, 0.001))
    )
    # naive_cal_level_all = np.hstack(
    #     (naive_cal_level_all, np.arange(0.9991, 0.9999, 0.0001))
    # )
    # naive_cal_level_all = np.hstack(
    #     (naive_cal_level_all, np.arange(0.99991, 0.99999, 0.00001))
    # )
    # naive_cal_level_all = np.hstack(
    #     (naive_cal_level_all, np.arange(0.999991, 0.999999, 0.000001))
    # )
    prev_empirical_coverage = 0
    for naive_cal_level in naive_cal_level_all:
        cfg.score_method = 'naive'
        cfg.naive_cal_level = float(naive_cal_level)
        cfg.temperature_scaling = tc
        agent = ConformalPredictor(cfg)
        agent.calibrate(log=False)
        prediction_set_size, empirical_coverage, _, _ = agent.test()
        if empirical_coverage > target_coverage:
            break
        prev_empirical_coverage = empirical_coverage
    logging.info(
        'Naive cal level: {} for alpha {}. Empirical coverage: {}'.format(
            naive_cal_level, cfg.alpha, empirical_coverage
        )
    )

    # conformal
    cfg.score_method = 'conformal'
    cfg.temperature_scaling = tc
    agent = ConformalPredictor(cfg)
    agent.calibrate(log=False)
    # prediction_set_size, empirical_coverage, _, _ = agent.test()

    # manually go through the test set and find the failure examples
    data_all_cp_succeed_naive_fail = []
    data_all_cp_fail_naive_succeed = []
    data_all_cp_avoid_help = []
    cp_empirical_coverage = 0
    for data in agent.test_data:

        # Get information
        data['prompt'] = data['context'
                             ]  # TODO (repo-wise): do not use context as key
        true_label = data['true_label']
        top_tokens = data['top_tokens']
        top_logprobs = data['top_logprobs']

        # temperature scaling and get softmax
        top_smx = temperature_scaling(top_logprobs, cfg.temperature_scaling)
        data['top_smx'] = top_smx
        data['tc'] = tc

        # Get prediction set
        cp_prediction_set = get_prediction_set(
            top_tokens, top_smx, agent.qhat, 'conformal', cfg
        )
        naive_prediction_set = get_prediction_set(
            top_tokens, top_smx, agent.qhat, 'naive', cfg
        )
        data['cp_prediction_set'] = cp_prediction_set
        data['naive_prediction_set'] = naive_prediction_set

        # Check help and success
        cp_flag_help, cp_flag_success = check_help_success(
            cfg, cp_prediction_set
        )
        naive_flag_help, naive_flag_success = check_help_success(
            cfg, naive_prediction_set
        )
        data['cp_flag_help'] = cp_flag_help
        data['cp_flag_success'] = cp_flag_success
        data['naive_flag_help'] = naive_flag_help
        data['naive_flag_success'] = naive_flag_success

        # Check when CP succeeds but Naive fails - this is when Naive misses the true label, but CP includes it with a larger prediction set
        if cp_flag_success and not naive_flag_success:
            data_all_cp_succeed_naive_fail.append(data)

        # Check when CP fails but Naive succeeds
        if not cp_flag_success and naive_flag_success:
            data_all_cp_fail_naive_succeed.append(data)

        # Check when Naive includes additional option while CP give a singleton of true label
        if len(naive_prediction_set) > len(
            cp_prediction_set
        ) and true_label in cp_prediction_set and not cp_flag_help:
            # assert not cp_flag_help
            assert naive_flag_help
            data_all_cp_avoid_help.append(data)

        # Check CP empirical coverage - make sure it is the same as Naive
        if true_label in cp_prediction_set:
            cp_empirical_coverage += 1 / len(agent.test_data)
    logging.info('CP empirical coverage: {}'.format(cp_empirical_coverage))

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