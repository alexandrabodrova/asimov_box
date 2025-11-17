""" Get the conditional coverage bound of conformal prediction, with the beta distribution.

Reference: https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/correctness_checks.ipynb

"""
import os
import argparse
import random
from omegaconf import OmegaConf
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, betabinom
import seaborn as sns
from agent.predict.util import temperature_scaling


sns.set(font_scale=3)
sns.set_style('darkgrid', {'axes.linewidth': 3, 'axes.edgecolor': 'black'})


def main(cfg):

    # Load data
    with open(cfg.load_data_path, 'rb') as handle:
        data_all = pickle.load(handle)

    # Figure out number of calibration data
    num_data = len(data_all)
    if 'num_calibration_data' in cfg:
        n = cfg.num_calibration_data
    else:
        n = int(num_data * cfg.calibration_ratio)

    # instead, we fix conditional guarantee, n, and delta, then find alpha
    for alpha in np.arange(0.50, 0., -0.005):
        l = np.floor((n+1) * alpha)
        a = n + 1 - l
        b = l
        conditional_coverage = beta.ppf(cfg.delta, a, b)
        if conditional_coverage > 0.75:
            break
    print(alpha, conditional_coverage)

    # or, we fix conditional guarantee, alpha, and delta, then find n
    for n in np.arange(200, 1500, 20):
        l = np.floor((n+1) * 0.10)
        a = n + 1 - l
        b = l
        conditional_coverage = beta.ppf(cfg.delta, a, b)
        if conditional_coverage > 0.85:
            break
    print(n, conditional_coverage)

    # Conditional conformal is a beta distribution. Use inverse CDF/quantile to find the conditional coverage given the number of calibration data (n) and tail distribution parameter (delta)
    l = np.floor((n+1) * cfg.alpha)
    a = n + 1 - l
    b = l
    x = np.linspace(0.700, 0.975, 1000)
    rv = beta(a, b)
    conditional_coverage = beta.ppf(cfg.delta, a, b)

    # # conditional coverage from offset
    # offset = np.sqrt(-np.log(cfg.delta) / (2*n))
    # conditional_coverage_offset = 1 - cfg.alpha - offset
    # conditional_coverage_offset_2 = 1 - cfg.alpha - np.sqrt(
    #     -2 * cfg.alpha * np.log(cfg.delta) / n
    # ) + 2 * np.log(cfg.delta) / n

    # plot
    plt.figure(figsize=(20, 16))
    plt.plot(x, rv.pdf(x), lw=8, label=f'n={n}')
    plt.axvline(
        x=conditional_coverage, linestyle='--', color='g', linewidth=10,
        label=f'Conditional coverage={conditional_coverage:.3f}'
    )
    plt.title(f'PDF of beta distribution with delta={cfg.delta}')
    plt.xlabel('Coverage')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(
        os.path.join(cfg.save_dir, f'beta_distribution_delta_{cfg.delta}.png')
    )

    # Split data
    calib_ind_all = random.sample(range(num_data), n)
    calibration_data = [data_all[ind] for ind in calib_ind_all]
    test_data = [
        data_all[ind] for ind in range(num_data) if ind not in calib_ind_all
    ]

    # Get softmax scores of true labels in the calibration set, get score function
    cal_scores = []
    num_calib = 0
    for data in calibration_data:
        top_tokens = data['top_tokens']
        top_logprobs = data['top_logprobs']
        true_label = data['true_label']

        # For multi-label setting: choose the true label one with the highest score
        top_smx = temperature_scaling(top_logprobs, cfg.temperature_scaling)
        if cfg.multi_label:
            assert len(true_label) > 0  # should be a list
            true_label_smx = [
                top_smx[sig_ind]
                for sig_ind, sig in enumerate(top_tokens)
                if sig in true_label
            ]
            true_label = true_label[np.argmax(true_label_smx)]
        else:
            # human provides a list - remove list
            if isinstance(true_label, list):
                assert len(true_label) == 1
                true_label = true_label[0]

        # Do not use bad data
        if true_label not in top_tokens:
            continue
        num_calib += 1

        # score: 1 - softmax(true label)
        cal_scores.append(
            1 - np.exp(top_logprobs[top_tokens.index(true_label)])
        )

    # Define empirical quantile
    q_level = np.ceil((n+1) * (1 - cfg.alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')

    # Calculate prediction set for test data
    prediction_set_size = []
    num_correct_prediction_set = 0
    num_test = 0
    for data in test_data:
        top_tokens = data['top_tokens']
        top_logprobs = data['top_logprobs']
        true_label = data['true_label']

        # For multi-label setting: choose the true label one with the highest score
        top_smx = temperature_scaling(top_logprobs, cfg.temperature_scaling)
        if cfg.multi_label:
            assert len(true_label) > 0  # should be a list
            true_label_smx = [
                top_smx[sig_ind]
                for sig_ind, sig in enumerate(top_tokens)
                if sig in true_label
            ]
            true_label = true_label[np.argmax(true_label_smx)]
        else:
            # human provides a list - remove list
            if isinstance(true_label, list):
                assert len(true_label) == 1
                true_label = true_label[0]

        # Do not use bad data
        if true_label not in top_tokens:
            continue
        num_test += 1

        # include all choices with softmax score >= 1-qhat
        prediction_set = [
            token for token_ind, token in enumerate(top_tokens)
            if np.exp(top_logprobs[token_ind]) >= 1 - qhat
        ]
        prediction_set_size.append(len(prediction_set))

        # check if true label is in prediction set
        if not set(data['true_label']).isdisjoint(prediction_set):
            num_correct_prediction_set += 1

    # Calculate empirical coverage
    empirical_coverage = num_correct_prediction_set / num_test

    # Report
    print()
    print('============== Summary ==============')
    print('Number of calibration data: ', num_calib)
    print('Number of test data: ', num_test)
    print('Quantile value: ', qhat)
    print('Average prediction set size: ', np.mean(prediction_set_size))
    print('Marginal coverage: ', 1 - cfg.alpha)
    print('Conditional coverage: ', conditional_coverage)
    # print('Conditional coverage (offset): ', conditional_coverage_offset)
    # print('Conditional coverage (offset 2): ', conditional_coverage_offset_2)
    print('With probability: ', 1 - cfg.delta)
    print('Empirical coverage:', empirical_coverage)
    print(
        f'This means, with probability {1-cfg.delta}, the empirical coverage {empirical_coverage:.3f} is greater than conditional coverage {conditional_coverage:.3f}.'
    )
    print('The above is not always satisfied since our test data is finite.')
    # print('Estimated probability: ', np.mean(empirical_coverage_all > conditional_coverage))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)

    main(cfg)
