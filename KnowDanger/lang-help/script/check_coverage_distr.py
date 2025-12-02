""" Run conformal prediction for multiple times, check the distribution of empirical coverage.

"""
import os
import argparse
import pickle
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(font_scale=2.5)
sns.set_style('darkgrid', {'axes.linewidth': 3, 'axes.edgecolor': 'black'})

from predict.conformal import run_prediction

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
        cfg.alpha = 0.15  # coverage level
        cfg.load_data_path = 'data/v2_template/answer/lm_answer.pkl'
        cfg.save_dir = 'data/v2_template/calibrate'
        cfg.calibration_ratio = 0.8  # ratio of data used for calibration
        cfg.score_method = 'conformal'  # ['naive', 'conformal', 'conformal_top_k', 'adaptive_conformal', 'regularized_adaptive_conformal']
        cfg.temperature_scaling = 3
        if cfg.score_method == 'regularized_adaptive_conformal':
            cfg.k_reg = 1  # (larger lam_reg and smaller k_reg leads to smaller sets)
            cfg.lam_reg = 1e-4
            cfg.disallow_zero_sets = False  # Set this to False in order to see the coverage upper bound hold
            cfg.rand = True  # Set this to True in order to see the coverage upper bound hold
        elif cfg.score_method == 'naive':
            cfg.naive_cal_level = 0.75
        cfg.help_mode = 'from_all_mc'  # from_prediction_set

        cfg.num_seed = 100
    else:
        cfg = OmegaConf.load(args.cfg_file)

    # run with different seeds
    empirical_coverage_all = []
    for seed in range(cfg.num_seed):
        cfg.seed = seed
        _, empirical_coverage, _, _ = run_prediction(cfg)
        empirical_coverage_all.append(empirical_coverage)

    # Get number of calibration and test data - yikes
    with open(cfg.load_data_path, 'rb') as handle:
        data_all = pickle.load(handle)
    num_data = len(data_all)
    n = int(num_data * cfg.calibration_ratio)

    # Summary
    print()
    print('============== Summary ==============')
    print('Number of calibration data: ', n)
    print('Number of test data: ', num_data - n)
    print('Marginal coverage: ', 1 - cfg.alpha)
    print('Average empirical coverage: ', np.mean(empirical_coverage_all))

    # plot
    plt.figure(figsize=(20, 16))
    plt.hist(empirical_coverage_all)
    plt.axvline(
        x=1 - cfg.alpha, linestyle='--', color='r',
        label=f'Marginal coverage={1-cfg.alpha:.3f}'
    )
    plt.title(
        f'Histogram of empirical coverage of running {cfg.num_seed} seed'
    )
    plt.xlabel('Empirical coverage')
    plt.ylabel('Frequency')
    plt.legend()
    # plt.show()
    plt.savefig(
        os.path.join(
            cfg.save_dir, f'hist_empirical_coverage_{cfg.score_method}.png'
        )
    )
