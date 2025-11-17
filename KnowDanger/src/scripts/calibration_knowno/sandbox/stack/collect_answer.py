""" Multi-step-multi-label, human clarification, stacking, tabletop manipulation environment

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
import logging
import random
import numpy as np
from omegaconf import OmegaConf


def main(cfg):
    # Current step
    cur_step = cfg.current_step

    # Load answer data
    mc_post_data_path = os.path.join(
        cfg.parent_data_folder, cfg.mc_post_data_path_from_parent
    )
    with open(mc_post_data_path, 'rb') as f:
        data_all = pickle.load(f)

    # Load answer response data
    mc_post_response_data_path = os.path.join(
        cfg.parent_data_folder, cfg.mc_post_response_data_path_from_parent
    )
    with open(mc_post_response_data_path, 'rb') as f:
        mc_post_response_data_all = pickle.load(f)

    # Generate data
    answer_data_all = []
    for data_ind, (data, mc_post_response_data) in enumerate(
        zip(data_all, mc_post_response_data_all)
    ):

        # Extract
        top_logprobs_full = mc_post_response_data['response_full']["choices"][
            0]["logprobs"]["top_logprobs"][0]
        top_tokens = [token.strip() for token in top_logprobs_full.keys()]
        top_logprobs = [value for value in top_logprobs_full.values()]

        # Determine action
        if cfg.calibration_mode:  # choose the label from the true label with the highest logprob - for calibration, we do not care if the top logprob is a true label
            true_label_all = data[f'step_{cur_step}']['true_label']
            if true_label_all == ['E']:
                action = data[f'step_{cur_step}']['true_action'
                                                 ]  # human labeled
            else:
                true_label_ind_all = [
                    top_tokens.index(label) for label in true_label_all
                ]
                true_label_logprob_all = [
                    top_logprobs[ind] for ind in true_label_ind_all
                ]
                true_label = true_label_all[np.argmax(true_label_logprob_all)]
                action = data[f'step_{cur_step}']['mc_all'][
                    cfg.mc_action_pair[true_label]]

            # extract the object and location from e.g., "get_obj_pos('green triangle') and place at get_pos(l2)"
            # TODO: remove the quote around obj and loc
            obj = action.split('(')[1].split(')')[0]
            loc = action.split('(')[2].split(')')[0]
        else:
            raise NotImplementedError

        # Save
        data['action'].append([obj, loc])
        data[f'step_{cur_step}']['top_tokens'] = top_tokens
        data[f'step_{cur_step}']['top_logprobs'] = top_logprobs

        # Save
        answer_data_all.append(data)

    # Save
    with open(cfg.data_save_path, 'wb') as f:
        pickle.dump(answer_data_all, f)

    # Summary
    logging.info('\n============== Summary ==============')
    logging.info(f'Number of questions generated: {cfg.num_data}')
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
    cfg.parent_data_folder = os.path.dirname(cfg.data_folder)

    # Merge with prompt cfg
    cfg_prompt = OmegaConf.load(
        os.path.join(cfg.parent_data_folder, 'prompt.yaml')
    )
    cfg = OmegaConf.merge(cfg_prompt, cfg)

    # Merge with common cfg
    cfg_common = OmegaConf.load(
        os.path.join(cfg.parent_data_folder, 'common.yaml')
    )
    cfg = OmegaConf.merge(cfg_common, cfg)

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
