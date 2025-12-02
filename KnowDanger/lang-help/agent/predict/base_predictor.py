""" Base class of all prediction models. """

import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(font_scale=2.5)
sns.set_style('darkgrid', {'axes.linewidth': 3, 'axes.edgecolor': 'black'})


class BasePredictor:

    def __init__(self, cfg):

        self.cfg = cfg

        # Load data
        with open(cfg.load_data_path, 'rb') as handle:
            data_all = pickle.load(handle)

        # Split data in calibration and test set - randomly sample
        self.num_data = len(data_all)
        if cfg.calibration_ratio is not None:
            self.n = int(self.num_data * cfg.calibration_ratio)
            random.seed(cfg.seed)
            calib_ind_all = random.sample(range(self.num_data), self.n)
            self.calibration_data = [data_all[ind] for ind in calib_ind_all]
            self.test_data = [
                data_all[ind]
                for ind in range(self.num_data)
                if ind not in calib_ind_all
            ]
        else:
            # use first n data as calibration set
            self.n = cfg.num_calibration_data
            self.calibration_data = data_all[:self.n]
            self.test_data = data_all[self.n:]

    def test(self):
        pass

    def plot_score_histogram(self, scores, cfg, suffix=''):
        plt.figure(figsize=(20, 16))
        plt.hist(scores, bins=50, edgecolor='k', linewidth=1)
        plt.axvline(
            x=self.qhat, linestyle='--', color='r', label='Quantile value'
        )
        plt.title(
            'Histogram of softmax scores of true labels in the calibration set'
        )
        plt.xlabel('Score (1 - softmax(true label))')
        plt.legend()
        # plt.show()
        plt.savefig(
            os.path.join(
                cfg.save_dir,
                f'score_hist_alpha-{cfg.alpha}_tc-{cfg.temperature_scaling}'
                + suffix + '.png'
            )
        )

    def plot_prediction_set_size(self, prediction_set_size, cfg, suffix=''):
        plt.figure(figsize=(20, 16))
        plt.hist(
            prediction_set_size, bins=np.arange(-1, 5) + 0.5, edgecolor='k',
            linewidth=1
        )
        ax = plt.gca()
        ax.locator_params(integer=True)
        plt.title('Histogram of prediction set size')
        plt.xlabel('Prediction set size')
        plt.ylabel('Frequency')
        # plt.show()
        plt.savefig(
            os.path.join(
                cfg.save_dir,
                f'prediction_set_size_hist_alpha-{cfg.alpha}_tc-{cfg.temperature_scaling}'
                + suffix + '.png'
            )
        )

    def plot_softmax_residual(self, residual):
        plt.figure(figsize=(20, 16))
        plt.hist(residual, bins=20, edgecolor='k', linewidth=1)
        plt.axvline(
            x=1 - self.qhat, linestyle='--', color='r', label='Quantile value'
        )
        plt.title('Histogram of softmax residual')
        plt.legend()
        # plt.show()