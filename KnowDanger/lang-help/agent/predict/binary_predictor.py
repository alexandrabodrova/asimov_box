""" Run prediction with binary uncertainty (True/False)

"""
import os
import argparse
from omegaconf import OmegaConf
import numpy as np
import logging
from agent.predict.base_predictor import BasePredictor
from agent.predict.util import get_prediction_set, temperature_scaling


class BinaryPredictor(BasePredictor):

    def __init__(self, cfg):
        super().__init__(cfg)

    def calibrate(self, plot_fig=False, log=True):
        print("No need for calibration!")
        return

    def test(self, plot_fig=False, log=False):
        cfg = self.cfg
        prediction_set_size = []
        num_help = 0
        num_success = 0
        num_correct_prediction_set = 0

        # Loop over test data
        for data in self.test_data:

            # Get information
            true_label = data['true_label']
            top_tokens = data['top_tokens']
            top_logprobs = data['top_logprobs']
            top_logprobs = [float(logprob) for logprob in top_logprobs]
            top_smx = temperature_scaling(
                top_logprobs, cfg.temperature_scaling
            )

            # Get prediction set
            prediction_set = get_prediction_set(
                top_tokens, top_smx, None, cfg.score_method, cfg
            )

            # ask for help if prediction set is not a singleton or 'e/E' is in the prediction set
            flag_help = 'False' in prediction_set
            num_help += flag_help

            # check success
            if flag_help:
                num_success += 1
            else:
                num_success += true_label == 'True'

            # count
            if true_label in prediction_set:
                num_correct_prediction_set += 1
            prediction_set_size += [len(prediction_set)]

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
            logging.info('Empirical coverage: %.3f', empirical_coverage)
            logging.info('Number of help: %d', num_help)
            logging.info('Number of success: %d', num_success)
            logging.info('Help frequency: %.3f', help_freq)
            logging.info('Success ratio: %.3f', success_ratio)

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
    agent = BinaryPredictor(cfg)

    # Calibrate
    agent.calibrate(plot_fig=cfg.plot_fig)

    # Test - often we only calibrate
    if cfg.test:
        agent.test(plot_fig=cfg.plot_fig, log=(cfg.logging_path is not None))
