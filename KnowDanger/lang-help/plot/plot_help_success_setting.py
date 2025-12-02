""" Compare help frequency and success ratio of different methods, in the single-step setting.

Use run_prediction function from predict/conformal.py.

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

    # with open('help_att.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # with open('help_num.pkl', 'rb') as f:
    #     data = pickle.load(f)
    with open('help_spa.pkl', 'rb') as f:
        data = pickle.load(f)

    conformal_help_freq_avg = np.array(data['conformal_help_freq_all'])
    conformal_success_ratio_avg = np.array(data['conformal_success_ratio_all'])
    naive_help_freq_avg = np.array(data['naive_help_freq_all'])
    naive_success_ratio_avg = np.array(data['naive_success_ratio_all'])
    ensemble_help_freq_avg = np.array(data['ensemble_help_freq_all'])
    ensemble_success_ratio_avg = np.array(data['ensemble_success_ratio_all'])
    binary_help_freq = data['binary_help_freq']
    binary_success_ratio = data['binary_success_ratio']
    set_help_freq = data['set_help_freq']
    set_success_ratio = data['set_success_ratio']
    no_help_freq = data['no_help_freq']
    no_help_success_ratio = data['no_help_success_ratio']

    # Plot comparison
    style = ['o-', '^-', '*-']
    plt.figure(figsize=(10, 8))
    plt.plot(
        conformal_help_freq_avg[0],
        conformal_success_ratio_avg[0],
        style[0],
        alpha=1,
        color=np.array([241, 141, 0]) / 255,
        markersize=5,
        linewidth=7,
        label=f'KnowNo',
        zorder=3,
    )

    # add the two points for binary and prompt set
    plt.scatter(
        set_help_freq,
        set_success_ratio,
        s=2000,
        marker='*',
        # color=np.array([188, 189, 33]) / 255,
        color=np.array([137, 138, 12]) / 255,
        label='Prompt Set',
        zorder=5,
    )

    plt.plot(
        naive_help_freq_avg[0],
        naive_success_ratio_avg[0],
        style[0],
        alpha=1,
        color=np.array([31, 119, 180]) / 255,
        markersize=5,
        linewidth=8,
        label=f'Simple Set',
        zorder=2,
    )

    plt.scatter(
        binary_help_freq,
        binary_success_ratio,
        s=2000,
        marker='*',
        # color=np.array([21, 190, 207]) / 255,
        color=np.array([4, 130, 143]) / 255,
        label='Binary',
        zorder=6,
    )

    plt.plot(
        ensemble_help_freq_avg[0],
        ensemble_success_ratio_avg[0],
        style[0],
        alpha=1,
        color=np.array([128, 128, 128]) / 255,
        markersize=5,
        linewidth=7,
        label='Ensemble Set',
        zorder=1,
    )

    plt.scatter(
        no_help_freq,
        no_help_success_ratio,
        s=2000,
        marker='*',
        color=np.array([79, 72, 117]) / 255,
        label='No Help',
        zorder=6,
    )

    # add legend below figure in one row
    # plt.legend(
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, -0.2),
    #     ncol=3,
    #     fancybox=True,
    #     shadow=False,
    #     fontsize=50,
    #     columnspacing=0.7,
    #     # box off
    #     frameon=False,
    # )

    # remove background color
    ax = plt.gca()
    ax.set_facecolor('white')

    plt.xlim([0.0 - 0.01, 1.0 + 0.01])
    plt.ylim([0.5 - 0.01, 1.0 + 0.01])

    # # x ticks off
    # # plt.xticks([])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    # # y label ticks off
    # ax = plt.gca()
    # ax.tick_params(labelleft=False)

    # show small ticks
    ax.tick_params(which='minor', length=10, width=10, color='black')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(which="both", bottom=True, left=True)

    plt.savefig(
        'help-success-spa.png',
        dpi=300,
        bbox_inches='tight',
    )