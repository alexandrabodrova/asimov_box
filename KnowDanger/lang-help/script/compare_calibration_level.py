""" Compare the calibration threshold vs. empirical coverage for different methods.

"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import seaborn as sns
from agent.predict.conformal_predictor import ConformalPredictor


sns.set(font_scale=2.5)
sns.set_style('darkgrid', {'axes.linewidth': 3, 'axes.edgecolor': 'black'})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)

    # temperature scaling
    tc_all = cfg.temperature_scaling_all

    # conformal
    cp_alpha_all = np.arange(0.01, 0.5 + 0.01, 0.01)
    cp_prediction_set_size_all = []
    cp_empirical_coverage_all = []
    for tc in tc_all:
        cp_prediction_set_size_tmp_all = []
        cp_empirical_coverage_tmp_all = []
        for alpha in cp_alpha_all:
            cfg.score_method = 'conformal'
            cfg.alpha = float(alpha)
            cfg.temperature_scaling = tc
            agent = ConformalPredictor(cfg)
            agent.calibrate()
            prediction_set_size, empirical_coverage, _, _ = agent.test()
            cp_prediction_set_size_tmp_all.append(prediction_set_size)
            cp_empirical_coverage_tmp_all.append(empirical_coverage)
        cp_prediction_set_size_all.append(cp_prediction_set_size_tmp_all)
        cp_empirical_coverage_all.append(cp_empirical_coverage_tmp_all)

    # Regularized adaptive conformal
    raps_alpha_all = np.arange(0.01, 0.5 + 0.01, 0.01)
    raps_prediction_set_size_all = []
    raps_empirical_coverage_all = []
    for tc in tc_all:
        raps_prediction_set_size_temp_all = []
        raps_empirical_coverage_temp_all = []
        for alpha in raps_alpha_all:
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
            1 - np.array(cp_alpha_all),
            cp_empirical_coverage_all[i],
            style[0],
            alpha=1 - i / len(tc_all),
            color='#66c56c',
            markersize=5,
            linewidth=5,
            label='LABEL, TC={}'.format(tc_all[i]),
        )
    for i in range(len(tc_all)):
        plt.plot(
            1 - np.array(raps_alpha_all),
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
            naive_cal_level_all,
            naive_empirical_coverage_all[i],
            style[0],
            alpha=1 - i / len(tc_all),
            color='#ff6555',
            markersize=5,
            linewidth=5,
            label='Naive, TC={}'.format(tc_all[i]),
        )
    plt.legend(loc='lower right')
    # plt.xlim([1.0 - 0.1, 3.5 + 0.1])
    # plt.ylim([0.6 - 0.01, 1.0 + 0.01])
    plt.xlabel('Calibration level')
    plt.ylabel('Empirical coverage')
    tc_all_str = ''.join([str(tc) + '-' for tc in tc_all])[:-1]
    plt.savefig(os.path.join(cfg.save_dir, f'threshold_tc-{tc_all_str}.png'))
