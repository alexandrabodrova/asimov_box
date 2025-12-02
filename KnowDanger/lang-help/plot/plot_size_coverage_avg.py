""" Compare average prediction set size and empirical coverage for different methods.

Use run_prediction function from conformal.py.

"""
import os
import argparse
from omegaconf import OmegaConf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(font_scale=4)
sns.set_style(
    'darkgrid',
    {
        'axes.linewidth': 3,
        'axes.edgecolor': 'black'
    },
)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-cf", "--cfg_file", help="cfg file path", default='', type=str
    # )
    # args = parser.parse_args()
    # cfg = OmegaConf.load(args.cfg_file)
    # cfg.save_dir = os.path.dirname(args.cfg_file)

    # # temperature scaling
    # tc_all = cfg.temperature_scaling_all

    with open('size_att.pkl', 'rb') as f:
        size_att = pickle.load(f)
    with open('size_num.pkl', 'rb') as f:
        size_num = pickle.load(f)
    with open('size_spa.pkl', 'rb') as f:
        size_spa = pickle.load(f)

    conformal_prediction_set_size_avg = (np.array(size_att['conformal_prediction_set_size_all']) + \
        np.array(size_num['conformal_prediction_set_size_all']) + \
        np.array(size_spa['conformal_prediction_set_size_all']))/3
    conformal_empirical_coverage_avg = (np.array(size_att['conformal_empirical_coverage_all']) + \
        np.array(size_num['conformal_empirical_coverage_all']) + \
        np.array(size_spa['conformal_empirical_coverage_all']))/3
    naive_prediction_set_size_avg = (np.array(size_att['naive_prediction_set_size_all']) + \
        np.array(size_num['naive_prediction_set_size_all']) + \
        np.array(size_spa['naive_prediction_set_size_all']))/3
    naive_empirical_coverage_avg = (np.array(size_att['naive_empirical_coverage_all']) + \
        np.array(size_num['naive_empirical_coverage_all']) + \
        np.array(size_spa['naive_empirical_coverage_all']))/3
    ensemble_prediction_set_size_avg = (np.array(size_att['ensemble_prediction_set_size_all']) + \
        np.array(size_num['ensemble_prediction_set_size_all']) + \
        np.array(size_spa['ensemble_prediction_set_size_all']))/3
    ensemble_empirical_coverage_avg = (np.array(size_att['ensemble_empirical_coverage_all']) + \
        np.array(size_num['ensemble_empirical_coverage_all']) + \
        np.array(size_spa['ensemble_empirical_coverage_all']))/3
    set_prediction_set_size_avg = (size_att['set_prediction_set_size'] + \
        size_num['set_prediction_set_size'] + \
        size_spa['set_prediction_set_size'])/3
    set_empirical_coverage_avg = (size_att['set_empirical_coverage'] + \
        size_num['set_empirical_coverage'] + \
        size_spa['set_empirical_coverage'])/3
    no_help_prediction_set_size_avg = (
        size_att['no_help_prediction_set_size']
        + size_num['no_help_prediction_set_size']
        + size_spa['no_help_prediction_set_size']
    ) / 3
    no_help_empirical_coverage_avg = (
        size_att['no_help_empirical_coverage']
        + size_num['no_help_empirical_coverage']
        + size_spa['no_help_empirical_coverage']
    ) / 3

    # Plot comparison
    style = ['o-', '^-', '*-']  # f4b247
    plt.figure(figsize=(10, 8))
    plt.plot(
        conformal_prediction_set_size_avg[0],
        conformal_empirical_coverage_avg[0],
        style[0],
        alpha=1,
        color=np.array([241, 141, 0]) / 255,
        # color=np.array((50, 205, 50)) / 255,
        # color=np.array((60, 179, 113)) / 255,
        markersize=5,
        linewidth=7,
        label='KnowNo',
        zorder=3
    )
    plt.plot(
        # naive_cal_level_all,
        naive_prediction_set_size_avg[0],
        naive_empirical_coverage_avg[0],
        style[0],
        alpha=1,
        # color='#ff6555',
        # color=np.array([76, 124, 133]) / 255,
        color=np.array([31, 119, 180]) / 255,
        markersize=5,
        linewidth=7,
        label='Simple Set',
        zorder=2,
    )
    plt.plot(
        # naive_cal_level_all,
        ensemble_prediction_set_size_avg[0],
        ensemble_empirical_coverage_avg[0],
        style[0],
        alpha=1,
        # color='#23aaff',
        color=np.array([128, 128, 128]) / 255,
        markersize=5,
        linewidth=7,
        label='Ensemble Set',
        zorder=1,
    )

    # add the two points for binary and prompt set
    # plt.scatter(
    #     binary_prediction_set_size,
    #     binary_empirical_coverage,
    #     s=20,
    #     color='#f4b247',
    #     label='Binary',
    # )
    plt.scatter(
        set_prediction_set_size_avg,
        set_empirical_coverage_avg,
        s=2000,
        marker='*',
        # color='#D8BFD8',
        zorder=4,
        # color='b',
        # color=np.array([188, 189, 33]) / 255,
        color=np.array([137, 138, 12]) / 255,
        label='Prompt Set',
    )

    # remove background color
    ax = plt.gca()
    ax.set_facecolor('white')

    # ax = plt.gca()
    # ax.tick_params(labelleft=False)

    plt.xticks([1, 2, 3, 4])
    plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])

    # show small ticks
    ax.tick_params(which='minor', length=10, width=10, color='black')
    # ax.tick_params(axis='x', which='minor', bottom=False)

    # plt.legend(loc='lower right')
    plt.xlim([1.0 - 0.1, 4.0 + 0.1])
    plt.ylim([0.6 - 0.01, 1.0 + 0.01])
    # plt.xlabel('Calibration level')
    # plt.xlabel('Average prediction set size')
    # plt.ylabel('Empirical coverage')
    # plt.show()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which="both", bottom=True, left=True)

    plt.savefig(
        'size.png',
        dpi=300,
        bbox_inches='tight',
    )
