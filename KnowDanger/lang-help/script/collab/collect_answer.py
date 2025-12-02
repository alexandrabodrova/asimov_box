""" Multi-step-multi-label, human collaboration, tabletop manipulation environment

Combine data with LM response to the multiple choices, for conformal prediction.

At calibration time, we assume the each step succeeds, and since this is the multi-step setting, we choose the action executed as the true label with the highest logprob.

At test time, there are three scenarios:
1. prediction set singleton and contains one of the true label, human does not help and succeeds.
2. prediction set does not contain any of the true label, human does not help and fails.
3. prediction set not singleton and contains at least one of the true label, human helps by choosing the label in the prediction set with the highest logprob, and succeeds.

"""
import os
import argparse
import pickle
import random
import logging
import numpy as np
from omegaconf import OmegaConf
from util.collab import check_true_label, get_true_action


def main(cfg):
    # Current step
    cur_step = cfg.current_step

    # Load previous data
    prev_data_path = os.path.join(
        cfg.parent_data_folder, cfg.prev_data_path_from_parent
    )
    with open(prev_data_path, 'rb') as f:
        data_all = pickle.load(f)

    # Load answer response data
    mc_post_response_data_path = os.path.join(
        cfg.parent_data_folder, cfg.mc_post_response_data_path_from_parent
    )
    if cfg.use_palm:
        with open(mc_post_response_data_path, 'r') as f:
            mc_post_response_data_all = f.read().split('\n')
    else:
        with open(mc_post_response_data_path, 'rb') as f:
            mc_post_response_data_all = pickle.load(f)

    # Generate data
    answer_data_all = []
    num_none_label = 0
    for data_ind, (data, mc_post_response_data) in enumerate(
        zip(data_all, mc_post_response_data_all)
    ):

        # Extract
        if cfg.use_palm:
            logprob_split = mc_post_response_data.split(' ')
            if len(logprob_split) < 5 or any(
                logprob == '' for logprob in logprob_split
            ):
                flag_rerun = True
                top_logprobs = None
                raise 'Missing scores - rerun the data'
            else:
                flag_rerun = False
                top_logprobs = [float(v) for v in logprob_split]
            top_tokens = ['A', 'B', 'C', 'D', 'E']
        else:
            top_logprobs_full = mc_post_response_data['response_full'][
                "choices"][0]["logprobs"]["top_logprobs"][0]
            top_tokens = [token.strip() for token in top_logprobs_full.keys()]
            top_logprobs = [value for value in top_logprobs_full.values()]

        # Determine true label
        true_labels = []
        true_mc_all = []
        none_index = None
        mc_all = data[f'step_{cur_step}']['mc_all']
        for i, mc in enumerate(mc_all):
            if mc == 'do nothing':
                continue
            if 'not listed here' in mc:
                none_index = i
                none_sig = top_tokens[i]
                continue
            if len(mc) < 5:
                continue

            #
            if 'm&m' in mc:
                mc = mc.replace('m&m', 'M&M')
            if 'skittles' in mc:
                mc = mc.replace('skittles', 'Skittles')

            # Check if true label
            true_label_mc = check_true_label(data, mc)
            if true_label_mc == 'True':
                true_labels.append(top_tokens[i])
                true_mc_all.append(mc)
        if len(true_labels) == 0:
            true_labels = [top_tokens[none_index]]
            num_none_label += 1

        # Determine action
        if cfg.calibration_mode:  # choose the label from the true label with the highest logprob - for calibration, we do not care if the top logprob is a true label
            if true_labels == [none_sig]:
                obj_true, loc_true = random.choice(get_true_action(data))
            else:
                true_label_inds = [
                    top_tokens.index(label) for label in true_labels
                ]
                true_label_logprobs = [
                    top_logprobs[ind] for ind in true_label_inds
                ]
                best_label = true_labels[np.argmax(true_label_logprobs)]
                true_action = mc_all[top_tokens.index(best_label)]

                # extract the object and location from e.g., "get_obj_pos('green triangle') and place at get_pos(l2)"
                obj_true = true_action.split(' in')[0].split('put ')[1]
                loc_true = ' '.join(true_action.split(' ')[-2:])
                if 'plate' not in loc_true:
                    breakpoint()
        else:
            raise NotImplementedError

        # Save
        data['action'].append([obj_true, loc_true])
        data[f'step_{cur_step}']['top_tokens'] = top_tokens
        data[f'step_{cur_step}']['top_logprobs'] = top_logprobs
        data[f'step_{cur_step}']['true_label'] = true_labels
        data[f'step_{cur_step}']['add_mc_prefix'] = none_index
        data['top_tokens'] = top_tokens
        data['top_logprobs'] = top_logprobs
        data['true_label'] = true_labels
        data['add_mc_prefix'] = none_index

        # Save
        answer_data_all.append(data)

    # Save
    with open(cfg.data_save_path, 'wb') as f:
        pickle.dump(answer_data_all, f)

    # Summary
    logging.info('\n============== Summary ==============')
    logging.info(f'Number of questions generated: {cfg.num_data}')
    logging.info(
        f'Number of questions with None option label: {num_none_label}'
    )
    logging.info(f'Data saved to: {cfg.data_save_path}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)
    cfg.data_folder = os.path.dirname(args.cfg_file)
    cfg.parent_data_folder = os.path.dirname(os.path.dirname(cfg.data_folder))

    # Logging
    cfg.logging_path = os.path.join(
        cfg.data_folder, cfg.log_file_name + '.log'
    )
    logging.basicConfig(
        level=logging.INFO, format='%(message)s', handlers=[
            logging.FileHandler(cfg.logging_path, mode='w'),
            logging.StreamHandler()
        ]
    )

    # Save path
    cfg.data_save_path = os.path.join(
        cfg.data_folder, cfg.data_save_name + '.pkl'
    )

    # run
    random.seed(cfg.seed)
    main(cfg)