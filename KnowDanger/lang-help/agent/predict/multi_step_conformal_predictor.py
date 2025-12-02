""" Run conformal prediction for multi-step template data (i.e., multiple choices generated with templates, instead of by prompting LM), by lifting to sequences.

The results include (1) average prediction set size, (2) empirical coverage over steps, (3) empirical coverage over trials, (4) help frequency over steps, (5) help frequency over trials, and (6) success ratio over trials.

"""
import os
import argparse
import logging
import numpy as np
from omegaconf import OmegaConf
from agent.predict.conformal_predictor import ConformalPredictor
from agent.predict.util import get_score, get_prediction_set, temperature_scaling


class MultiStepConformalPredictor(ConformalPredictor):

    def __init__(self, cfg):
        super().__init__(cfg)

        # number of steps in each trial
        self.num_step = cfg.num_step

    def calibrate(self, plot_fig=False):
        cfg = self.cfg
        cal_scores_trial = []
        cal_scores_steps = {step + 1: [] for step in range(self.num_step)}
        for data in self.calibration_data:

            # store scores of each step in one trial
            score_steps = []

            # flag for bad data
            flag_bad_data = False

            # each step
            for step in range(self.num_step):

                # extract data
                true_label = data[f'step_{step+1}']['true_label']
                assert type(true_label) == list
                top_tokens = data[f'step_{step+1}']['top_tokens']
                top_logprobs = data[f'step_{step+1}']['top_logprobs']

                # Do not use bad data
                if set(true_label).isdisjoint(set(top_tokens)):
                    flag_bad_data = True
                    raise

                # temperature scaling and get softmax
                if 'top_smx' in data:  # from ensemble
                    top_smx = np.array(data[f'step_{step+1}']['top_smx'])
                else:
                    top_smx = temperature_scaling(
                        top_logprobs, cfg.temperature_scaling
                    )

                # For multi-label setting: choose the true label one with the highest score
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

                # Get calibration score
                score = get_score(
                    top_tokens,
                    top_smx,
                    true_label,
                    cfg.score_method,
                    cfg,
                )
                score_steps.append(score)
                cal_scores_steps[step + 1].append(score)

            # maximum score (i.e., minimum confidence) of all steps
            cal_score = max(score_steps)

            # store score of the trial if not bad data
            cal_scores_trial += [cal_score] if not flag_bad_data else []

        # Define empirical quantile for trial-level score
        self.qhat = np.quantile(
            cal_scores_trial, self.q_level, method='higher'
        )

        # Plot score histogram
        if plot_fig:
            self.plot_score_histogram(cal_scores_trial, cfg)
            for step in range(self.num_step):
                self.plot_score_histogram(
                    cal_scores_steps[step + 1], cfg,
                    suffix='_step_{}'.format(step + 1)
                )

        # Report
        logging.info('============== Summary ==============')
        logging.info('Number of calibration data: %d', self.n)
        logging.info('Quantile level: %.3f', self.q_level)
        logging.info('Quantile value: %.5f', self.qhat)

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
            flag_trial_helped = False

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
                prediction_set = get_prediction_set(
                    top_tokens, top_smx, self.qhat, cfg.score_method, cfg
                )

                # ask for help -  no need if the trial already fails at previous step
                if not flag_trial_fail:

                    # ask for help if prediction set is not a singleton or 'e/E' is in the prediction set
                    flag_step_helped = False
                    if cfg.count_e_as_help:
                        if len(
                            prediction_set
                        ) != 1 or none_option_token in prediction_set:
                            num_step_helped += 1
                            flag_step_helped = True
                            flag_trial_helped = True
                    else:
                        if len(prediction_set) != 1:
                            num_step_helped += 1
                            flag_step_helped = True
                            flag_trial_helped = True

                    # check success
                    if flag_step_helped:
                        if len(prediction_set) == 1:
                            assert prediction_set[0] == none_option_token
                            continue
                        elif cfg.help_mode == 'from_prediction_set':
                            if set(true_label).isdisjoint(prediction_set):
                                flag_trial_fail = True
                        elif cfg.help_mode == 'from_all_mc':  # always work
                            pass
                        else:
                            raise 'Unknown help mode!'
                    else:
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

                # Log if specified
                # if log and set(true_label).isdisjoint(prediction_set):
                #     # if step == 2:
                #     #     for obj in data['init']['objects']:
                #     #         if obj not in fail_obj_count:
                #     #             fail_obj_count[obj] = 0
                #     #         fail_obj_count[obj] += 1
                #     # find object in the true label
                #     for label in true_label:
                #         mc = data['step_2']['mc_all'][cfg.mc_sigs.index(label)]
                #         try:
                #             obj_mc = mc.split(' in')[0].split('put ')[1]
                #             # print(obj_mc)
                #         except:
                #             # breakpoint()
                #             continue
                #         if obj_mc not in fail_obj_count:
                #             fail_obj_count[obj_mc] = 0
                #         fail_obj_count[obj_mc] += 1

                # if step == 2:
                #     logging.info(
                #         '----------------------------------------'
                #     )
                #     logging.info(data[f'step_{step+1}']['mc_post_prompt'])
                #     logging.info([
                #         f'prob: {np.exp(top_logprobs[top_tokens.index(sig)]):.3f}'
                #         for sig in cfg.mc_sigs
                #     ])
                #     logging.info(
                #         f'True label: {true_label}; Predition set: {prediction_set}'
                #     )
                #     logging.info(
                #         '----------------------------------------\n'
                #     )
                #     input()

            # Summarize trial result
            num_trial_helped += 1 if flag_trial_helped else 0
            num_trial_success += 1 if not flag_trial_fail else 0
            num_trial_with_correct_prediction_set += 0 if flag_trial_with_incorrect_prediction_set else 1

        # plot fail object count in a bar plot
        fail_obj_count = sorted(
            fail_obj_count.items(), key=lambda x: x[1], reverse=True
        )
        # print(fail_obj_count)

        # Plot prediction set histogram - wider is better
        if plot_fig:
            self.plot_prediction_set_size(prediction_set_size, cfg)
            for step in range(self.num_step):
                self.plot_prediction_set_size(
                    prediction_set_size_step[step + 1], cfg,
                    suffix='_step_{}'.format(step + 1)
                )

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
        # logging.info(f'Quantile level: {q_level}')
        # logging.info(f'Quantile value: {self.qhat}')
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
    agent = MultiStepConformalPredictor(cfg)

    # Calibrate
    agent.calibrate(plot_fig=cfg.plot_fig)

    # Test - often we only calibrate
    if cfg.test:
        agent.test(plot_fig=cfg.plot_fig, log=(cfg.logging_path is not None))
