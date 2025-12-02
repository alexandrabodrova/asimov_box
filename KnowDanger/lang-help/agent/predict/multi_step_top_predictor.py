""" 
"""
import os
import argparse
import logging
import numpy as np
from omegaconf import OmegaConf
from agent.predict.top_predictor import TopPredictor
from agent.predict.util import get_score, get_prediction_set, temperature_scaling


class MultiStepTopPredictor(TopPredictor):

    def __init__(self, cfg):
        super().__init__(cfg)

        # number of steps in each trial
        self.num_step = cfg.num_step

    def calibrate(self, plot_fig=False):
        pass

    def test(self, plot_fig=False, log=False):
        cfg = self.cfg
        num_test_data = self.num_data - self.n
        prediction_set_size = []
        prediction_set_size_step = {
            step + 1: [] for step in range(self.num_step)
        }
        num_trial_with_correct_prediction_set = 0
        num_step_with_correct_prediction_set = 0
        num_trial_helped = 0
        num_step_helped = 0
        num_trial_success = 0
        num_step = 0  # since some steps, fail before end

        fail_obj_count = {}

        # Loop over test data
        for data in self.test_data:

            # flag for whether the trial fails and whether the prediction set is correct
            flag_trial_fail = False
            flag_trial_with_incorrect_prediction_set = False

            # each step
            for step in range(self.num_step):
                num_step += 1

                # extract data
                true_label = data[f'step_{step+1}']['true_label']
                assert type(true_label) == list
                top_tokens = data[f'step_{step+1}']['top_tokens']
                top_logprobs = data[f'step_{step+1}']['top_logprobs']
                if 'add_mc_prefix' in data:
                    none_option_token = data['add_mc_prefix']
                else:
                    none_option_token = cfg.e_sig  # default to 'e/E'

                # temperature scaling and get softmax
                if 'top_smx' in data:  # from ensemble
                    top_smx = np.array(data[f'step_{step+1}']['top_smx'])
                else:
                    top_smx = temperature_scaling(
                        top_logprobs, cfg.temperature_scaling
                    )

                # Get prediction set
                prediction_set = [top_tokens[np.argmax(top_smx)]]

                # ask for help -  no need if the trial already fails at previous step
                if not flag_trial_fail:
                    assert len(prediction_set) == 1
                    if set(true_label).isdisjoint(prediction_set):
                        flag_trial_fail = True

                # Check coverage
                if not set(true_label).isdisjoint(prediction_set):
                    num_step_with_correct_prediction_set += 1
                else:
                    flag_trial_with_incorrect_prediction_set = True

                # Count prediction set size
                prediction_set_size.append(len(prediction_set))
                prediction_set_size_step[step + 1].append(len(prediction_set))

            # Summarize trial result
            num_trial_success += 1 if not flag_trial_fail else 0
            num_trial_with_correct_prediction_set += 0 if flag_trial_with_incorrect_prediction_set else 1

        # plot fail object count in a bar plot
        fail_obj_count = sorted(
            fail_obj_count.items(), key=lambda x: x[1], reverse=True
        )
        # print(fail_obj_count)

        # Summarize results
        avg_prediction_set_size = np.mean(prediction_set_size)
        empirical_coverage_over_step = num_step_with_correct_prediction_set / (
            num_test_data * self.num_step
        )
        empirical_coverage_over_trial = num_trial_with_correct_prediction_set / num_test_data
        help_freq_over_step = num_step_helped / num_step
        help_freq_over_trial = num_trial_helped / num_test_data
        success_ratio = num_trial_success / num_test_data

        # Report
        logging.info('\n============== Summary ==============')
        logging.info(f'Number of calibration data: {self.n}')
        logging.info(f'Number of test data: {num_test_data}')
        logging.info(
            f'Average prediction set size: {np.mean(prediction_set_size)}'
        )
        logging.info(f'Marginal coverage (over trials): {1-cfg.alpha}')
        logging.info(
            f'Empirical coverage (over trials): {empirical_coverage_over_trial}'
        )
        logging.info(
            f'Empirical coverage (over steps): {empirical_coverage_over_step}'
        )
        logging.info(f'Number of steps helped: {num_step_helped}')
        logging.info(f'Number of trials helped: {num_trial_helped}')
        logging.info(f'Number of successful trials: {num_trial_success}')
        logging.info(f'Help frequency (over trials): {help_freq_over_trial}')
        logging.info(f'Help frequency (over steps): {help_freq_over_step}')
        logging.info(f'Success ratio: {success_ratio}')

        return avg_prediction_set_size, empirical_coverage_over_step, empirical_coverage_over_trial, help_freq_over_step, help_freq_over_trial, success_ratio


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
    agent = MultiStepTopPredictor(cfg)

    # Test
    agent.test(plot_fig=cfg.plot_fig, log=(cfg.logging_path is not None))
