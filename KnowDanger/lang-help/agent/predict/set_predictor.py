""" Run prediction with binary uncertainty (True/False)

"""
import os
import argparse
from omegaconf import OmegaConf
import numpy as np
import logging
from agent.predict.base_predictor import BasePredictor


class SetPredictor(BasePredictor):

    def __init__(self, cfg):
        super().__init__(cfg)

    def calibrate(self, plot_fig=False, log=True):
        print("No need for calibration!")
        return

    def test(self, plot_fig=False, log=False):
        cfg = self.cfg
        prediction_set_size = []
        num_correct_prediction_set = 0
        num_help = 0
        num_success = 0
        smx_residual_all = []

        # Loop over test data
        for data in self.test_data:

            # Get information
            true_label = data['true_label']
            top_tokens = data['top_tokens']
            if 'add_mc_prefix' in data:
                none_option_token = data['add_mc_prefix']
            else:
                none_option_token = cfg.e_sig  # default to 'e/E'

            # Make sure true label is a list
            if not isinstance(true_label, list):
                true_label = [true_label]

            # Get prediction set
            prediction_set = data['top_set']

            # ask for help if prediction set is not a singleton or 'e/E' is in the prediction set
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
            if flag_help:
                if cfg.help_mode == 'from_prediction_set':
                    num_success += not set(true_label
                                          ).isdisjoint(prediction_set)
                elif cfg.help_mode == 'from_all_mc':  # always work
                    num_success += 1
                else:
                    raise 'Unknown help model!'
            else:
                assert len(prediction_set) == 1
                num_success += not set(true_label).isdisjoint(prediction_set)

            # Check if true label is in prediction set
            num_correct_prediction_set += not set(true_label
                                                 ).isdisjoint(prediction_set)
            prediction_set_size.append(len(prediction_set))
            # prediction_set_size_per_type[true_type].append(len(prediction_set))

            # Log if specified
            # if log:
            #     logging.info('----------------------------------------')
            #     logging.info(data['request'])
            #     logging.info(data['mc_prompt'])
            #     logging.info(
            #         f'True label: {true_label}; Prediction set: {prediction_set}'
            #     )
            #     logging.info(
            #         f'Smx: {[np.round(top_smx[top_tokens.index(sig)], 3) for sig in cfg.mc_sigs if sig in top_tokens]}'
            #     )
            #     logging.info('----------------------------------------\n')

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
            # logging.info(
            #     'Average prediction set size per type (eq, amb, sem): %s', [
            #         np.mean(prediction_set_size_per_type[key])
            #         if len(prediction_set_size_per_type[key]) > 0 else 0
            #         for key in prediction_set_size_per_type.keys()
            #     ]
            # )
            # logging.info('Marginal coverage: %.3f', 1 - cfg.alpha)
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
    agent = SetPredictor(cfg)

    # Calibrate
    agent.calibrate(plot_fig=cfg.plot_fig)

    # Test - often we only calibrate
    if cfg.test:
        agent.test(plot_fig=cfg.plot_fig, log=(cfg.logging_path is not None))
