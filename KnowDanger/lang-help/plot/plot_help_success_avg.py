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

    with open('help_att.pkl', 'rb') as f:
        help_att = pickle.load(f)
    with open('help_num.pkl', 'rb') as f:
        help_num = pickle.load(f)
    with open('help_spa.pkl', 'rb') as f:
        help_spa = pickle.load(f)

    conformal_help_freq_avg = (np.array(help_att['conformal_help_freq_all']) + \
        np.array(help_num['conformal_help_freq_all']) + \
        np.array(help_spa['conformal_help_freq_all']))/3
    conformal_success_ratio_avg = (np.array(help_att['conformal_success_ratio_all']) + \
        np.array(help_num['conformal_success_ratio_all']) + \
        np.array(help_spa['conformal_success_ratio_all']))/3
    naive_help_freq_avg = (np.array(help_att['naive_help_freq_all']) + \
        np.array(help_num['naive_help_freq_all']) + \
        np.array(help_spa['naive_help_freq_all']))/3
    naive_success_ratio_avg = (np.array(help_att['naive_success_ratio_all']) + \
        np.array(help_num['naive_success_ratio_all']) + \
        np.array(help_spa['naive_success_ratio_all']))/3
    ensemble_help_freq_avg = (np.array(help_att['ensemble_help_freq_all']) + \
        np.array(help_num['ensemble_help_freq_all']) + \
        np.array(help_spa['ensemble_help_freq_all']))/3
    ensemble_success_ratio_avg = (np.array(help_att['ensemble_success_ratio_all']) + \
        np.array(help_num['ensemble_success_ratio_all']) + \
        np.array(help_spa['ensemble_success_ratio_all']))/3
    binary_help_freq = (help_att['binary_help_freq'] + \
        help_num['binary_help_freq'] + \
        help_spa['binary_help_freq'])/3
    binary_success_ratio = (help_att['binary_success_ratio'] + \
        help_num['binary_success_ratio'] + \
        help_spa['binary_success_ratio'])/3
    set_help_freq = (help_att['set_help_freq'] + \
        help_num['set_help_freq'] + \
        help_spa['set_help_freq'])/3
    set_success_ratio = (help_att['set_success_ratio'] + \
        help_num['set_success_ratio'] + \
        help_spa['set_success_ratio'])/3
    no_help_freq = (help_att['no_help_freq'] + \
        help_num['no_help_freq'] + \
        help_spa['no_help_freq'])/3
    no_help_success_ratio = (help_att['no_help_success_ratio'] + \
        help_num['no_help_success_ratio'] + \
        help_spa['no_help_success_ratio'])/3

    # import pickle
    # with open('help_spa.pkl', 'wb') as f:
    #     pickle.dump({
    #         'conformal_help_freq_all': conformal_help_freq_all,
    #         'naive_help_freq_all': naive_help_freq_all,
    #         'ensemble_help_freq_all': ensemble_help_freq_all,
    #         'set_help_freq': set_help_freq,
    #         'binary_help_freq': binary_help_freq,
    #         'conformal_success_ratio_all': conformal_success_ratio_all,
    #         'naive_success_ratio_all': naive_success_ratio_all,
    #         'ensemble_success_ratio_all': ensemble_success_ratio_all,
    #         'set_success_ratio': set_success_ratio,
    #         'binary_success_ratio': binary_success_ratio,
    #     }, f)

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
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        fancybox=True,
        shadow=False,
        fontsize=50,
        columnspacing=0.7,
        # box off
        frameon=False,
    )

    # remove background color
    ax = plt.gca()
    ax.set_facecolor('white')

    plt.xlim([0.0 - 0.01, 1.0 + 0.01])
    plt.ylim([0.6 - 0.01, 1.0 + 0.01])

    # # x ticks off
    # # plt.xticks([])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
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
        'help.png',
        dpi=300,
        bbox_inches='tight',
    )

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
