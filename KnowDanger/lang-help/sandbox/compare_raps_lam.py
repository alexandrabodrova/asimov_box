""" Compare RAPS with different regularization parameters (regularized adaptive conformal prediction).

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
    if args.cfg_file == '':
        print('Using pre-defined parameters!')
        cfg = OmegaConf.create()
        cfg.seed = 42
        cfg.alpha = 0.1  # coverage level
        cfg.calibration_ratio = 0.8
        cfg.load_data_path = 'data/v2_template/answer/lm_answer.pkl'
        cfg.save_dir = 'data/v2_template/calibrate'
        cfg.score_method = 'regularized_adaptive_conformal'  # ['naive', 'conformal', 'conformal_top_k', 'adaptive_conformal', 'regularized_adaptive_conformal']
        cfg.temperature_scaling = 10
        cfg.help_mode = 'from_all_mc'  # 'from_prediction_set'
    else:
        cfg = OmegaConf.load(args.cfg_file)

    # temperature scaling
    lam_all = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    # k_reg_all = [0,1,2,3,4]

    # conformal
    alpha_all = np.arange(0.05, 0.5 + 0.01, 0.004)
    raps_prediction_set_size_all = []
    raps_empirical_coverage_all = []
    for lam in lam_all:
        # for k_reg in k_reg_all:
        raps_prediction_set_size_temp_all = []
        raps_empirical_coverage_temp_all = []
        for alpha in alpha_all:
            cfg.score_method = 'regularized_adaptive_conformal'
            cfg.alpha = float(alpha)
            cfg.k_reg = 1
            cfg.lam_reg = lam
            cfg.disallow_zero_sets = False
            cfg.rand = True
            predictor = ConformalPredictor(cfg)
            predictor.calibrate()
            prediction_set_size, empirical_coverage, _, _ = predictor.test()
            raps_prediction_set_size_temp_all.append(prediction_set_size)
            raps_empirical_coverage_temp_all.append(empirical_coverage)
        raps_prediction_set_size_all.append(raps_prediction_set_size_temp_all)
        raps_empirical_coverage_all.append(raps_empirical_coverage_temp_all)

    # Plot comparison
    style = ['o-', '^-', '*-']  # f4b247
    plt.figure(figsize=(20, 16))
    for i in range(len(lam_all)):
        plt.plot(
            raps_prediction_set_size_all[i], raps_empirical_coverage_all[i],
            style[0], alpha=1 - i / len(lam_all), color='#23aaff',
            markersize=5, linewidth=5, label='RAPS, lam={}'.format(lam_all[i])
        )
    # plt.plot(conformal_prediction_set_size_all, conformal_empirical_coverage_all, 'o-', markersize=3, label='Conformal')
    # plt.plot(naive_prediction_set_size_all, naive_empirical_coverage_all, 'o-', markersize=3, label='Naive')
    plt.legend(loc='lower right')
    plt.xlim([1.0 - 0.1, 3.5 + 0.1])
    plt.ylim([0.7, 0.95])
    # plt.ylim([0.5, 0.9])
    plt.xlabel('Average prediction set size')
    plt.ylabel('Empirical coverage')
    # plt.show()
    plt.savefig(os.path.join(cfg.save_dir, 'rasp_lam.png'))
