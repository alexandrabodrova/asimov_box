""" Compare help frequency and success ratio of different methods, in the single-step setting.

Use run_prediction function from predict/conformal.py.

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
from agent.predict.binary_predictor import BinaryPredictor
from agent.predict.set_predictor import SetPredictor
from agent.predict.top_predictor import TopPredictor

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
    cfg.load_data_path = os.path.join(
        cfg.save_dir, 'collect_mc', 'answer', 'answer.pkl'
    )
    if 'alpha_all' in cfg:
        alpha_all = cfg.alpha_all
    else:
        alpha_all = np.arange(0.01, 0.25 + 0.01, 0.005)
    conformal_help_freq_all = []
    conformal_success_ratio_all = []
    for tc in tc_all:
        conformal_help_freq_tmp_all = []
        conformal_sucess_ratio_tmp_all = []
        for alpha in alpha_all:
            cfg.score_method = 'conformal'
            cfg.alpha = float(alpha)
            cfg.temperature_scaling = tc
            agent = ConformalPredictor(cfg)
            agent.calibrate()
            _, _, help_freq, success_ratio = agent.test()
            conformal_help_freq_tmp_all.append(help_freq)
            conformal_sucess_ratio_tmp_all.append(success_ratio)
        conformal_help_freq_all.append(conformal_help_freq_tmp_all)
        conformal_success_ratio_all.append(conformal_sucess_ratio_tmp_all)

    # naive
    if 'alpha_all' in cfg:
        naive_cal_level_all = [1 - alpha for alpha in cfg.alpha_all]
    else:
        naive_cal_level_all = np.arange(0.4, 0.99, 0.005)
    naive_help_freq_all = []
    naive_success_ratio_all = []
    for tc in tc_all:
        naive_help_freq_tmp_all = []
        naive_success_ratio_tmp_all = []
        for naive_cal_level in naive_cal_level_all:
            cfg.score_method = 'naive'
            cfg.naive_cal_level = float(naive_cal_level)
            cfg.temperature_scaling = tc
            agent = ConformalPredictor(cfg)
            agent.calibrate()
            _, _, help_freq, success_ratio = agent.test()
            naive_help_freq_tmp_all.append(help_freq)
            naive_success_ratio_tmp_all.append(success_ratio)
        naive_help_freq_all.append(naive_help_freq_tmp_all)
        naive_success_ratio_all.append(naive_success_ratio_tmp_all)

    # ensemble
    cfg.load_data_path = os.path.join(
        cfg.save_dir, 'collect_ensemble', 'answer_few', 'answer.pkl'
    )
    naive_cal_level_all = np.arange(0.4, 0.99, 0.01)
    ensemble_help_freq_all = []
    ensemble_success_ratio_all = []
    for tc in tc_all:
        ensemble_help_freq_tmp_all = []
        ensemble_success_ratio_tmp_all = []
        for naive_cal_level in naive_cal_level_all:
            cfg.score_method = 'naive'
            cfg.naive_cal_level = float(naive_cal_level)
            cfg.temperature_scaling = tc
            agent = ConformalPredictor(cfg)
            agent.calibrate()
            _, _, help_freq, success_ratio = agent.test()
            ensemble_help_freq_tmp_all.append(help_freq)
            ensemble_success_ratio_tmp_all.append(success_ratio)
        ensemble_help_freq_all.append(ensemble_help_freq_tmp_all)
        ensemble_success_ratio_all.append(ensemble_success_ratio_tmp_all)

    # binary
    cfg.load_data_path = os.path.join(
        cfg.save_dir, 'collect_binary', 'answer', 'answer.pkl'
    )
    cfg.naive_cal_level = 0.5
    agent = BinaryPredictor(cfg)
    agent.calibrate()
    _, _, binary_help_freq, binary_success_ratio = agent.test()

    # prompt set
    cfg.load_data_path = os.path.join(
        cfg.save_dir, 'collect_set', 'answer', 'answer.pkl'
    )
    agent = SetPredictor(cfg)
    agent.calibrate()
    _, _, set_help_freq, set_success_ratio = agent.test()

    # no-help
    cfg.load_data_path = os.path.join(
        cfg.save_dir, 'collect_mc', 'answer', 'answer.pkl'
    )
    agent = TopPredictor(cfg)
    # agent.calibrate()
    _, _, no_help_freq, no_help_success_ratio = agent.test()

    import pickle
    with open('help_spa.pkl', 'wb') as f:
        pickle.dump({
            'conformal_help_freq_all': conformal_help_freq_all,
            'naive_help_freq_all': naive_help_freq_all,
            'ensemble_help_freq_all': ensemble_help_freq_all,
            'set_help_freq': set_help_freq,
            'binary_help_freq': binary_help_freq,
            'conformal_success_ratio_all': conformal_success_ratio_all,
            'naive_success_ratio_all': naive_success_ratio_all,
            'ensemble_success_ratio_all': ensemble_success_ratio_all,
            'set_success_ratio': set_success_ratio,
            'binary_success_ratio': binary_success_ratio,
            'no_help_freq': no_help_freq,
            'no_help_success_ratio': no_help_success_ratio,
        }, f)

    # # Plot comparison
    # style = ['o-', '^-', '*-']
    # plt.figure(figsize=(10, 8))
    # for i in range(len(tc_all)):
    #     plt.plot(
    #         conformal_help_freq_all[i],
    #         conformal_success_ratio_all[i],
    #         style[0],
    #         alpha=1 - i / len(tc_all),
    #         color=np.array([241, 141, 0]) / 255,
    #         markersize=5,
    #         linewidth=8,
    #         label=f'KnowNo',
    #         zorder=3,
    #     )
    # for i in range(len(tc_all)):
    #     plt.plot(
    #         naive_help_freq_all[i],
    #         naive_success_ratio_all[i],
    #         style[0],
    #         alpha=1 - i / len(tc_all),
    #         color=np.array([76, 124, 133]) / 255,
    #         markersize=5,
    #         linewidth=8,
    #         label=f'Simple Set',
    #         zorder=2,
    #     )
    # for i in range(len(tc_all)):
    #     plt.plot(
    #         ensemble_help_freq_all[i],
    #         ensemble_success_ratio_all[i],
    #         style[0],
    #         alpha=1 - i / len(tc_all),
    #         color=np.array([128, 128, 128]) / 255,
    #         markersize=5,
    #         linewidth=8,
    #         label='Ensemble Set',
    #         zorder=1,
    #     )

    # # add the two points for binary and prompt set
    # plt.scatter(
    #     set_help_freq,
    #     set_success_ratio,
    #     s=800,
    #     marker='*',
    #     color='#D8BFD8',
    #     label='Prompt Set',
    #     zorder=5,
    # )
    # plt.scatter(
    #     binary_help_freq,
    #     binary_success_ratio,
    #     s=800,
    #     marker='*',
    #     color=np.array([167, 167, 167]) / 255,
    #     label='Binary',
    #     zorder=4,
    # )

    # # add legend below figure in one row
    # plt.legend(
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, -0.2),
    #     ncol=5,
    #     fancybox=True,
    #     shadow=False,
    #     fontsize=50,
    #     columnspacing=0.7,
    #     # box off
    #     frameon=False,
    # )

    # # remove background color
    # ax = plt.gca()
    # ax.set_facecolor('white')

    # plt.xlim([0.0 - 0.01, 1.0 + 0.01])
    # plt.ylim([0.6 - 0.01, 1.0 + 0.01])

    # # x ticks off
    # # plt.xticks([])
    # plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])

    # # y label ticks off
    # ax = plt.gca()
    # ax.tick_params(labelleft=False)

    # # ax.tick_params(axis='x', which='major', pad=-10)

    # # plt.xlabel('Help frequency')
    # # plt.ylabel('Success ratio')
    # # plt.show()
    # tc_all_str = ''.join([str(tc) + '-' for tc in tc_all])[:-1]
    # plt.savefig(
    #     os.path.join(cfg.save_dir, f'help_success_tc-{tc_all_str}.png'),
    #     dpi=300,
    #     bbox_inches='tight',
    # )
