""" Compare average prediction set size and empirical coverage for different methods.

Use run_prediction function from conformal.py.

"""
import os
import argparse
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(font_scale=2.5)
sns.set_style('darkgrid', {'axes.linewidth': 3, 'axes.edgecolor': 'black'})

from agent.predict.conformal_predictor import ConformalPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    cfg.save_dir = os.path.dirname(args.cfg_file)

    # temperature scaling
    tc_all = cfg.temperature_scaling_all

    # conformal
    alpha_all = np.arange(0.01, 0.3 + 0.01, 0.03)
    conformal_prediction_set_size_all = []
    conformal_empirical_coverage_all = []
    for tc in tc_all:
        conformal_prediction_set_size_temp_all = []
        conformal_empirical_coverage_temp_all = []
        for alpha in alpha_all:
            cfg.score_method = 'conformal'
            cfg.alpha = float(alpha)
            cfg.temperature_scaling = tc
            agent = ConformalPredictor(cfg)
            agent.calibrate()
            prediction_set_size, empirical_coverage, _, _ = agent.test()
            conformal_prediction_set_size_temp_all.append(prediction_set_size)
            conformal_empirical_coverage_temp_all.append(empirical_coverage)
        conformal_prediction_set_size_all.append(
            conformal_prediction_set_size_temp_all
        )
        conformal_empirical_coverage_all.append(
            conformal_empirical_coverage_temp_all
        )

    # Regularized adaptive conformal
    alpha_all = np.arange(0.01, 0.3 + 0.01, 0.03)
    raps_prediction_set_size_all = []
    raps_empirical_coverage_all = []
    for tc in tc_all:
        raps_prediction_set_size_temp_all = []
        raps_empirical_coverage_temp_all = []
        for alpha in alpha_all:
            cfg.score_method = 'regularized_adaptive_conformal'
            cfg.alpha = float(alpha)
            cfg.temperature_scaling = tc
            cfg.k_reg = 1
            cfg.lam_reg = 1e-4
            cfg.disallow_zero_sets = False
            cfg.rand = True
            agent = ConformalPredictor(cfg)
            agent.calibrate()
            prediction_set_size, empirical_coverage, _, _ = agent.test()
            raps_prediction_set_size_temp_all.append(prediction_set_size)
            raps_empirical_coverage_temp_all.append(empirical_coverage)
        raps_prediction_set_size_all.append(raps_prediction_set_size_temp_all)
        raps_empirical_coverage_all.append(raps_empirical_coverage_temp_all)

    # naive
    naive_cal_level_all = np.arange(0.4, 0.99, 0.03)
    naive_cal_level_all = np.hstack(
        (naive_cal_level_all, np.arange(0.991, 0.999, 0.001))
    )
    naive_cal_level_all = np.hstack(
        (naive_cal_level_all, np.arange(0.9991, 0.9999, 0.0001))
    )
    naive_cal_level_all = np.hstack(
        (naive_cal_level_all, np.arange(0.99991, 0.99999, 0.00001))
    )
    naive_cal_level_all = np.hstack(
        (naive_cal_level_all, np.arange(0.999991, 0.999999, 0.000001))
    )
    naive_prediction_set_size_all = []
    naive_empirical_coverage_all = []
    for tc in tc_all:
        naive_prediction_set_size_temp_all = []
        naive_empirical_coverage_temp_all = []
        for naive_cal_level in naive_cal_level_all:
            cfg.score_method = 'naive'
            cfg.naive_cal_level = float(naive_cal_level)
            cfg.temperature_scaling = tc
            agent = ConformalPredictor(cfg)
            agent.calibrate()
            prediction_set_size, empirical_coverage, _, _ = agent.test()
            naive_prediction_set_size_temp_all.append(prediction_set_size)
            naive_empirical_coverage_temp_all.append(empirical_coverage)
        naive_prediction_set_size_all.append(
            naive_prediction_set_size_temp_all
        )
        naive_empirical_coverage_all.append(naive_empirical_coverage_temp_all)

    # Plot comparison
    style = ['o-', '^-', '*-']  # f4b247
    plt.figure(figsize=(20, 16))
    for i in range(len(tc_all)):
        plt.plot(
            # 1 - np.array(alpha_all),
            conformal_prediction_set_size_all[i],
            conformal_empirical_coverage_all[i],
            style[0],
            alpha=1 - i / len(tc_all),
            color='#66c56c',
            markersize=5,
            linewidth=5,
            label='LABEL, TC={}'.format(tc_all[i]),
        )
    for i in range(len(tc_all)):
        plt.plot(
            raps_prediction_set_size_all[i],
            raps_empirical_coverage_all[i],
            style[0],
            alpha=1 - i / len(tc_all),
            color='#23aaff',
            markersize=5,
            linewidth=5,
            label='RAPS, TC={}'.format(tc_all[i]),
        )
    for i in range(len(tc_all)):
        plt.plot(
            # naive_cal_level_all,
            naive_prediction_set_size_all[i],
            naive_empirical_coverage_all[i],
            style[0],
            alpha=1 - i / len(tc_all),
            color='#ff6555',
            markersize=5,
            linewidth=5,
            label='Naive, TC={}'.format(tc_all[i]),
        )
    plt.legend(loc='lower right')
    plt.xlim([1.0 - 0.1, 3.5 + 0.1])
    plt.ylim([0.6 - 0.01, 1.0 + 0.01])
    # plt.xlabel('Calibration level')
    plt.xlabel('Average prediction set size')
    plt.ylabel('Empirical coverage')
    # plt.show()
    tc_all_str = ''.join([str(tc) + '-' for tc in tc_all])[:-1]
    plt.savefig(
        os.path.join(cfg.save_dir, f'size_coverage_tc-{tc_all_str}.png')
        # os.path.join(cfg.save_dir, f'cal_level_tc-{tc_all_str}.png')
    )
