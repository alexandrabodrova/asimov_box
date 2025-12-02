""" Always choose the top choice"""
import os
import argparse
import logging
import numpy as np
from omegaconf import OmegaConf
from agent.predict.base_predictor import BasePredictor
from agent.predict.util import temperature_scaling


class TopPredictor(BasePredictor):

    def __init__(self, cfg):
        super().__init__(cfg)

    def calibrate(self, plot_fig=False, log=True):
        pass

    def test(self, plot_fig=False, log=False):
        cfg = self.cfg
        prediction_set_size = []
        num_correct_prediction_set = 0
        num_help = 0
        num_success = 0
        num_weird = 0

        # Loop over test data
        for data in self.test_data:

            # Get information
            true_label = data['true_label']
            top_tokens = data['top_tokens']
            top_logprobs = data['top_logprobs']

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

            # Get prediction set - just the highest score one
            prediction_set = [top_tokens[np.argmax(top_smx)]]

            # no help
            flag_help = False

            # check success
            true_label_in_prediction_set = not set(true_label
                                                  ).isdisjoint(prediction_set)
            num_success += true_label_in_prediction_set

            # Check if true label is in prediction set
            num_correct_prediction_set += true_label_in_prediction_set
            prediction_set_size.append(len(prediction_set))

        # Plot prediction set histogram - wider is better
        if plot_fig:
            self.plot_prediction_set_size(prediction_set_size, cfg)

        # Summarize results
        avg_prediction_set_size = np.mean(prediction_set_size)
        empirical_coverage = num_correct_prediction_set / (
            self.num_data - self.n
        )
        help_freq = num_help / len(self.test_data)
        success_ratio = num_success / len(self.test_data)

        # Report
        if log:
            logging.info('============== Summary ==============')
            logging.info('Number of calibration data: %d', self.n)
            logging.info('Number of test data: %d', self.num_data - self.n)
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
            logging.info('Number of weird: %d', num_weird)

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
    agent = TopPredictor(cfg)

    # Test
    agent.test(plot_fig=cfg.plot_fig, log=(cfg.logging_path is not None))