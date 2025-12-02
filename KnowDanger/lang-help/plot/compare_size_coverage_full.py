""" Compare average prediction set size and empirical coverage for different methods.

Use run_prediction function from conformal.py.

"""
import os
import argparse
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(font_scale=6)
sns.set_style(
    'darkgrid',
    {
        'axes.linewidth': 3,
        'axes.edgecolor': 'black'
    },
)

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
    alpha_all = np.arange(0.005, 0.25 + 0.01, 0.005)
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

    # naive
    naive_cal_level_all = np.arange(0.4, 0.99, 0.005)
    # naive_cal_level_all = np.hstack(
    #     (naive_cal_level_all, np.arange(0.991, 0.999, 0.001))
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

    # ensemble
    cfg.load_data_path = os.path.join(
        cfg.save_dir, 'collect_ensemble', 'answer_few', 'answer.pkl'
    )
    naive_cal_level_all = np.arange(0.7, 0.99, 0.01)
    naive_cal_level_all = np.hstack(
        (naive_cal_level_all, np.arange(0.991, 0.999, 0.001))
    )
    ensemble_prediction_set_size_all = []
    ensemble_empirical_coverage_all = []
    for tc in tc_all:
        ensemble_prediction_set_size_temp_all = []
        ensemble_empirical_coverage_temp_all = []
        for naive_cal_level in naive_cal_level_all:
            cfg.score_method = 'naive'
            cfg.naive_cal_level = float(naive_cal_level)
            cfg.temperature_scaling = tc
            agent = ConformalPredictor(cfg)
            agent.calibrate()
            prediction_set_size, empirical_coverage, _, _ = agent.test()
            ensemble_prediction_set_size_temp_all.append(prediction_set_size)
            ensemble_empirical_coverage_temp_all.append(empirical_coverage)
        ensemble_prediction_set_size_all.append(
            ensemble_prediction_set_size_temp_all
        )
        ensemble_empirical_coverage_all.append(
            ensemble_empirical_coverage_temp_all
        )

    # binary
    cfg.load_data_path = os.path.join(
        cfg.save_dir, 'collect_binary', 'answer', 'answer.pkl'
    )
    agent = BinaryPredictor(cfg)
    agent.calibrate()
    binary_prediction_set_size, binary_empirical_coverage, _, _ = agent.test()

    # prompt set
    cfg.load_data_path = os.path.join(
        cfg.save_dir, 'collect_set', 'answer', 'answer.pkl'
    )
    agent = SetPredictor(cfg)
    agent.calibrate()
    set_prediction_set_size, set_empirical_coverage, _, _ = agent.test()

    # no-help
    cfg.load_data_path = os.path.join(
        cfg.save_dir, 'collect_mc', 'answer', 'answer.pkl'
    )
    agent = TopPredictor(cfg)
    # agent.calibrate()
    no_help_prediction_set_size, no_help_empirical_coverage, _, _ = agent.test(
    )

    import pickle
    with open('size_spa.pkl', 'wb') as f:
        pickle.dump({
            'conformal_prediction_set_size_all':
                conformal_prediction_set_size_all,
            'naive_prediction_set_size_all': naive_prediction_set_size_all,
            'ensemble_prediction_set_size_all':
                ensemble_prediction_set_size_all,
            'set_prediction_set_size': set_prediction_set_size,
            'no_help_prediction_set_size': no_help_prediction_set_size,
            'conformal_empirical_coverage_all':
                conformal_empirical_coverage_all,
            'naive_empirical_coverage_all': naive_empirical_coverage_all,
            'ensemble_empirical_coverage_all': ensemble_empirical_coverage_all,
            'set_empirical_coverage': set_empirical_coverage,
            'no_help_empirical_coverage': no_help_empirical_coverage,
        }, f)

    # Plot comparison
    # style = ['o-', '^-', '*-']  # f4b247
    # plt.figure(figsize=(10, 8))
    # for i in range(len(tc_all)):
    #     plt.plot(
    #         # 1 - np.array(alpha_all),
    #         conformal_prediction_set_size_all[i],
    #         conformal_empirical_coverage_all[i],
    #         style[0],
    #         alpha=1 - i / len(tc_all),
    #         # color='#66c56c',
    #         color=np.array([241, 141, 0]) / 255,
    #         # color=np.array((50, 205, 50)) / 255,
    #         # color=np.array((60, 179, 113)) / 255,
    #         markersize=5,
    #         linewidth=8,
    #         label='KnowNo',
    #         zorder=3,
    #     )
    # for i in range(len(tc_all)):
    #     plt.plot(
    #         # naive_cal_level_all,
    #         naive_prediction_set_size_all[i],
    #         naive_empirical_coverage_all[i],
    #         style[0],
    #         alpha=1 - i / len(tc_all),
    #         # color='#ff6555',
    #         color=np.array([76, 124, 133]) / 255,
    #         markersize=5,
    #         linewidth=8,
    #         label='Simple Set',
    #         zorder=2,
    #     )
    # for i in range(len(tc_all)):
    #     plt.plot(
    #         # naive_cal_level_all,
    #         ensemble_prediction_set_size_all[i],
    #         ensemble_empirical_coverage_all[i],
    #         style[0],
    #         alpha=1 - i / len(tc_all),
    #         # color='#23aaff',
    #         color=np.array([128, 128, 128]) / 255,
    #         markersize=5,
    #         linewidth=8,
    #         label='Ensemble Set',
    #         zorder=1,
    #     )

    # # add the two points for binary and prompt set
    # # plt.scatter(
    # #     binary_prediction_set_size,
    # #     binary_empirical_coverage,
    # #     s=20,
    # #     color='#f4b247',
    # #     label='Binary',
    # # )
    # plt.scatter(
    #     set_prediction_set_size,
    #     set_empirical_coverage,
    #     s=800,
    #     marker='*',
    #     color='#D8BFD8',
    #     zorder=4,
    #     # color='b',
    #     # color=np.array([0, 128, 128]) / 255,
    #     label='Prompt Set',
    # )

    # # remove background color
    # ax = plt.gca()
    # ax.set_facecolor('white')

    # # ax = plt.gca()
    # # ax.tick_params(labelleft=False)

    # plt.xticks([1, 2, 3, 4])

    # # plt.legend(loc='lower right')
    # plt.xlim([1.0 - 0.1, 4.0 + 0.1])
    # plt.ylim([0.6 - 0.01, 1.0 + 0.01])
    # # plt.xlabel('Calibration level')
    # # plt.xlabel('Average prediction set size')
    # # plt.ylabel('Empirical coverage')
    # # plt.show()
    # tc_all_str = ''.join([str(tc) + '-' for tc in tc_all])[:-1]
    # plt.savefig(
    #     os.path.join(cfg.save_dir, f'size_coverage_tc-{tc_all_str}.png'),
    #     dpi=300,
    #     bbox_inches='tight',
    # )
