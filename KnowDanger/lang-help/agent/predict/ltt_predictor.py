""" Learn then Test (LTT)

Control the expected failure rate when the prediction set is a singleton.

"""
import os
import argparse
import logging
import numpy as np
from omegaconf import OmegaConf
from agent.predict.base_predictor import BasePredictor
from agent.predict.util import get_score, get_prediction_set, temperature_scaling


class LttPredictor(BasePredictor):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.num_qhat = cfg.num_qhat
        self.alpha = cfg.alpha  # risk control
        self.delta = cfg.delta  # high-probability

    def calibrate(self, plot_fig=False, log=True):

        qhat = np.linspace(0, 1, self.num_qhat)
        losses = np.zeros((self.n, self.num_qhat)
                         )  # loss for example i with parameter qhat[j]
        num_good = 0
        num_all = 0
        for i, data in enumerate(self.calibration_data):

            # Get information
            true_label = data['true_label']
            top_tokens = data['top_tokens']
            top_logprobs = data['top_logprobs']

            # temperature scaling and get softmax
            if 'top_smx' in data:  # from ensemble
                top_smx = np.array(data['top_smx'])
            else:
                top_smx = temperature_scaling(
                    top_logprobs, cfg.temperature_scaling
                )

            for j in range(self.num_qhat):
                prediction_set = get_prediction_set(
                    top_tokens, top_smx, qhat[j], cfg.score_method, cfg
                )
                if len(prediction_set) == 1:
                    losses[i,
                           j] = set(true_label
                                   ).isdisjoint(prediction_set) + 1  # hacky

        def ReLU(x):
            return x * (x > 0)

        risk = []
        for j in range(self.num_qhat):
            if np.sum(losses[:, j]) >= 1:
                risk.append(
                    np.sum(losses[:, j] == 2) / np.sum(losses[:, j] >= 1)
                )
            else:
                risk.append(0)
        risk = np.array(risk)
        pvals = np.exp(
            -2 * self.n * (ReLU(self.alpha - risk)**2)
        )  # Or the HB p-value
        # Fixed-sequence test starting at qhat[-1] and ending at qhat[0]
        below_delta = pvals <= self.delta
        valid = np.array([
            (np.mean(below_delta[j:]) == 1) for j in range(self.num_qhat)
        ])
        self.qhat_valid = qhat[valid]

        # Report
        if log:
            logging.info('============== Summary ==============')
            logging.info('Number of calibration data: %d', self.n)
            logging.info(
                'Valid quantile value: %s', np.array2string(self.qhat_valid)
            )

    def test(self, plot_fig=False, log=False):
        """Use the smallest valid qhat"""
        self.qhat = self.qhat_valid[0]

        cfg = self.cfg
        prediction_set_size = []
        num_correct_prediction_set = 0
        num_help = 0
        num_success = 0
        num_singleton_set = 0
        num_singleton_set_success = 0

        # Loop over test data
        for data in self.test_data:

            # Get information
            true_label = data['true_label']
            top_tokens = data['top_tokens']
            top_logprobs = data['top_logprobs']
            if 'add_mc_prefix' in data:
                none_option_token = data['add_mc_prefix']
            else:
                none_option_token = cfg.e_sig  # default to 'e/E'

            # Make sure true label is a list
            if not isinstance(true_label, list):
                true_label = [true_label]

            # temperature scaling and get softmax
            if 'top_smx' in data:  # from ensemble
                top_smx = np.array(data['top_smx'])
            else:
                top_smx = temperature_scaling(
                    top_logprobs, cfg.temperature_scaling
                )

            # Get prediction set
            prediction_set = get_prediction_set(
                top_tokens, top_smx, self.qhat, cfg.score_method, cfg
            )
            if len(prediction_set) == 1:
                num_singleton_set += 1
                if not set(true_label).isdisjoint(prediction_set):
                    num_singleton_set_success += 1

            # ask for help if prediction set is (1) not a singleton or (2) equal to ['E']
            flag_help = False
            if cfg.count_e_as_help:
                cond = len(
                    prediction_set
                ) != 1 or none_option_token in prediction_set
            else:
                cond = len(prediction_set) != 1
            if cond:
                num_help += 1
                flag_help = True

            # check success
            true_label_in_prediction_set = not set(true_label
                                                  ).isdisjoint(prediction_set)
            if flag_help:
                # success if (1) true label is in prediction set (true label can also be E, then human provides aciton) or
                # (2) prediction set is empty (human provides aciton)
                if cfg.help_mode == 'from_prediction_set':
                    flag_success = true_label_in_prediction_set or len(
                        prediction_set
                    ) == 0
                elif cfg.help_mode == 'from_all_mc':  # always work
                    flag_success = True
                else:
                    raise 'Unknown help model!'
            else:
                assert len(prediction_set) == 1
                flag_success = true_label_in_prediction_set
            if flag_success:
                num_success += 1

            # Check if true label is in prediction set
            num_correct_prediction_set += true_label_in_prediction_set
            prediction_set_size.append(len(prediction_set))

        # Plot prediction set histogram - wider is better
        if plot_fig:
            self.plot_prediction_set_size(prediction_set_size, cfg)

        # counts of different set size from 0 to 5
        prediction_set_size_cnt = {
            '0': 0,
            '1': 0,
            '2': 0,
            '3': 0,
            '4': 0,
            '5': 0
        }
        for i in prediction_set_size:
            if i > 5:
                prediction_set_size_cnt['5'] += 1
            else:
                prediction_set_size_cnt[str(i)] += 1
        print(prediction_set_size_cnt)

        # Summarize results
        avg_prediction_set_size = np.mean(prediction_set_size)
        empirical_coverage = num_correct_prediction_set / (
            self.num_data - self.n
        )
        help_freq = num_help / len(self.test_data)
        success_ratio = num_success / len(self.test_data)
        no_help_success = num_singleton_set_success / num_singleton_set

        # Report
        if log:
            logging.info('============== Summary ==============')
            logging.info('Number of calibration data: %d', self.n)
            logging.info('Number of test data: %d', self.num_data - self.n)
            # logging.info('Quantile level: %.3f', q_level)
            logging.info('Quantile value: %.5f', self.qhat)
            logging.info(
                'Average prediction set size: %.3f',
                np.mean(prediction_set_size)
            )
            logging.info('Marginal coverage: %.3f', 1 - cfg.alpha)
            logging.info('Empirical coverage: %.3f', empirical_coverage)
            logging.info('Number of help: %d', num_help)
            logging.info('Number of success: %d', num_success)
            logging.info('Help frequency: %.3f', help_freq)
            logging.info('Success ratio: %.3f', success_ratio)
            logging.info('No help success ratio: %.3f', no_help_success)
        print(num_singleton_set_success, num_singleton_set)

        return avg_prediction_set_size, empirical_coverage, help_freq, success_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    cfg.save_dir = os.path.dirname(args.cfg_file)
    cfg.logging_path = os.path.join(cfg.save_dir, 'calibration.log')

    # logging
    if cfg.logging_path is not None:
        logging.basicConfig(
            level=logging.INFO, format='%(message)s', handlers=[
                logging.FileHandler(cfg.logging_path, mode='w'),
                logging.StreamHandler()
            ]
        )  # overwrite

    # Agent
    agent = LttPredictor(cfg)

    # Calibrate
    agent.calibrate(plot_fig=cfg.plot_fig)

    # Test - often we only calibrate
    if cfg.test:
        agent.test(plot_fig=cfg.plot_fig, log=(cfg.logging_path is not None))