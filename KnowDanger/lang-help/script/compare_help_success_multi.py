""" Compare help frequency and success ratio of different methods, in the multi-step setting.

Use run_prediction function from predict/conformal_multi.py.

"""
import os
import argparse
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(font_scale=2.5)
sns.set_style('darkgrid', {'axes.linewidth': 3, 'axes.edgecolor': 'black'})

from agent.predict.multi_step_conformal_predictor import MultiStepConformalPredictor

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
    alpha_all = np.arange(0.01, 0.5 + 0.01, 0.03)
    conformal_help_freq_over_step_all = []
    conformal_help_freq_over_trial_all = []
    conformal_success_ratio_all = []
    for tc in tc_all:
        conformal_help_freq_over_step_tmp_all = []
        conformal_help_freq_over_trial_tmp_all = []
        conformal_sucess_ratio_tmp_all = []
        for alpha in alpha_all:
            cfg.score_method = 'conformal'
            cfg.alpha = float(alpha)
            cfg.temperature_scaling = tc
            agent = MultiStepConformalPredictor(cfg)
            agent.calibrate()

            _, _, _, help_freq_over_step, help_freq_over_trial, success_ratio = agent.test(
            )

            conformal_help_freq_over_step_tmp_all.append(help_freq_over_step)
            conformal_help_freq_over_trial_tmp_all.append(help_freq_over_trial)
            conformal_sucess_ratio_tmp_all.append(success_ratio)
        conformal_help_freq_over_step_all.append(
            conformal_help_freq_over_step_tmp_all
        )
        conformal_help_freq_over_trial_all.append(
            conformal_help_freq_over_trial_tmp_all
        )
        conformal_success_ratio_all.append(conformal_sucess_ratio_tmp_all)

    # raps
    alpha_all = np.arange(0.01, 0.5 + 0.01, 0.03)
    raps_help_freq_over_step_all = []
    raps_help_freq_over_trial_all = []
    raps_success_ratio_all = []
    for tc in tc_all:
        raps_help_freq_over_step_tmp_all = []
        raps_help_freq_over_trial_tmp_all = []
        raps_sucess_ratio_tmp_all = []
        for alpha in alpha_all:
            cfg.score_method = 'regularized_adaptive_conformal'
            cfg.alpha = float(alpha)
            cfg.temperature_scaling = tc
            cfg.k_reg = 1
            cfg.lam_reg = 1e-4
            cfg.disallow_zero_sets = False
            cfg.rand = True
            agent = MultiStepConformalPredictor(cfg)
            agent.calibrate()

            _, _, _, help_freq_over_step, help_freq_over_trial, success_ratio = agent.test(
            )

            raps_help_freq_over_step_tmp_all.append(help_freq_over_step)
            raps_help_freq_over_trial_tmp_all.append(help_freq_over_trial)
            raps_sucess_ratio_tmp_all.append(success_ratio)
        raps_help_freq_over_step_all.append(raps_help_freq_over_step_tmp_all)
        raps_help_freq_over_trial_all.append(raps_help_freq_over_trial_tmp_all)
        raps_success_ratio_all.append(raps_sucess_ratio_tmp_all)

    # naive
    naive_cal_level_all = np.arange(0.3, 0.99, 0.03)
    # naive_cal_level_all = np.hstack(
    #     (naive_cal_level_all, np.arange(0.991, 0.999, 0.001))
    # )
    # naive_cal_level_all = np.hstack(
    #     (naive_cal_level_all, np.arange(0.9991, 0.9999, 0.0001))
    # )
    # naive_cal_level_all = np.hstack(
    #     (naive_cal_level_all, np.arange(0.99991, 0.99999, 0.00001))
    # )
    # naive_cal_level_all = np.hstack(
    #     (naive_cal_level_all, np.arange(0.999991, 0.999999, 0.000001))
    # )
    naive_help_freq_over_step_all = []
    naive_help_freq_over_trial_all = []
    naive_success_ratio_all = []
    for tc in tc_all:
        naive_help_freq_over_step_tmp_all = []
        naive_help_freq_over_trial_tmp_all = []
        naive_sucess_ratio_tmp_all = []
        for naive_cal_level in naive_cal_level_all:
            cfg.score_method = 'naive'
            cfg.naive_cal_level = float(naive_cal_level)
            cfg.temperature_scaling = tc
            agent = MultiStepConformalPredictor(cfg)
            agent.calibrate()

            _, _, _, help_freq_over_step, help_freq_over_trial, success_ratio = agent.test(
            )

            naive_help_freq_over_step_tmp_all.append(help_freq_over_step)
            naive_help_freq_over_trial_tmp_all.append(help_freq_over_trial)
            naive_sucess_ratio_tmp_all.append(success_ratio)
        naive_help_freq_over_step_all.append(naive_help_freq_over_step_tmp_all)
        naive_help_freq_over_trial_all.append(
            naive_help_freq_over_trial_tmp_all
        )
        naive_success_ratio_all.append(naive_sucess_ratio_tmp_all)

    # Plot with help frequency over trial
    style = ['o-', '^-', '*-']
    plt.figure(figsize=(20, 16))
    for i in range(len(tc_all)):
        plt.plot(
            conformal_help_freq_over_trial_all[i],
            conformal_success_ratio_all[i], style[0],
            alpha=1 - i / len(tc_all), color='#66c56c', markersize=5,
            linewidth=5, label=f'LABEL, TC={tc_all[i]}'
        )
    for i in range(len(tc_all)):
        plt.plot(
            raps_help_freq_over_trial_all[i], raps_success_ratio_all[i],
            style[0], alpha=1 - i / len(tc_all), color='#23aaff', markersize=5,
            linewidth=5, label=f'RAPS, TC={tc_all[i]}'
        )
    for i in range(len(tc_all)):
        plt.plot(
            naive_help_freq_over_trial_all[i], naive_success_ratio_all[i],
            style[0], alpha=1 - i / len(tc_all), color='#ff6555', markersize=5,
            linewidth=5, label=f'Naive, TC={tc_all[i]}'
        )
    plt.legend(loc='lower right')
    plt.xlim([0.0 - 0.01, 1.0 + 0.01])
    plt.ylim([0.5 - 0.01, 1.0 + 0.01])
    plt.xlabel('Help frequency (over trial)')
    plt.ylabel('Success ratio')
    # plt.show()
    plt.savefig(os.path.join(cfg.save_dir, 'help_success_trial.png'))

    # Plot with help frequency over step
    style = ['o-', '^-', '*-']
    plt.figure(figsize=(20, 16))
    for i in range(len(tc_all)):
        plt.plot(
            conformal_help_freq_over_step_all[i],
            conformal_success_ratio_all[i], style[0],
            alpha=1 - i / len(tc_all), color='#66c56c', markersize=5,
            linewidth=5, label=f'LABEL, TC={tc_all[i]}'
        )
    for i in range(len(tc_all)):
        plt.plot(
            raps_help_freq_over_step_all[i], raps_success_ratio_all[i],
            style[0], alpha=1 - i / len(tc_all), color='#23aaff', markersize=5,
            linewidth=5, label=f'RAPS, TC={tc_all[i]}'
        )
    for i in range(len(tc_all)):
        plt.plot(
            naive_help_freq_over_step_all[i], naive_success_ratio_all[i],
            style[0], alpha=1 - i / len(tc_all), color='#ff6555', markersize=5,
            linewidth=5, label=f'Naive, TC={tc_all[i]}'
        )
    plt.legend(loc='lower right')
    plt.xlim([0.0 - 0.01, 1.0 + 0.01])
    plt.ylim([0.5 - 0.01, 1.0 + 0.01])
    plt.xlabel('Help frequency (over step)')
    plt.ylabel('Success ratio')
    # plt.show()
    plt.savefig(os.path.join(cfg.save_dir, 'help_success_step.png'))
