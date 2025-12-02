""" Single-step, human clarification, tabletop manipulation environment

Combine data with LM response to the multiple choices, apply ensemble.

Prob(A) = occurence(A) / num_ensemble
Then we apply naive prediction set. Or use some threshold? top label prob < 0.5 for help?

"""
import os
import argparse
import pickle
import logging
import random
from omegaconf import OmegaConf
from util.collab import check_true_label, get_true_action


def main(cfg):
    # Current step
    cur_step = cfg.current_step

    # Load answer data
    prev_data_path = os.path.join(
        cfg.parent_data_folder, cfg.prev_data_path_from_parent
    )
    with open(prev_data_path, 'rb') as f:
        data_all = pickle.load(f)
    prev_data_path_1 = os.path.join(
        cfg.parent_data_folder, cfg.prev_data_path_from_parent_1
    )
    with open(prev_data_path_1, 'rb') as f:
        data_all_1 = pickle.load(f)
    prev_data_path_2 = os.path.join(
        cfg.parent_data_folder, cfg.prev_data_path_from_parent_2
    )
    with open(prev_data_path_2, 'rb') as f:
        data_all_2 = pickle.load(f)

    # Load answer response data
    mc_post_response_data_path = os.path.join(
        cfg.parent_data_folder, cfg.mc_post_response_data_path_from_parent
    )
    with open(mc_post_response_data_path, 'r') as f:
        mc_post_response_data_all = f.read().split('--0000--')

    # Generate data
    answer_data_all = []
    num_none_label = 0
    for data_ind, data in enumerate(data_all):
        data_1 = data_all_1[data_ind]
        data_2 = data_all_2[data_ind]

        # Use the first ensemble as mc_all
        mc_all = data['mc_post']['ens_0']['mc_all']
        mc_all_count_dict = {}
        for mc in mc_all:
            mc_all_count_dict[mc] = 0
        if len(
            list(mc_all_count_dict.keys())
        ) < 5:  # sometimes there are more than one 'do nothing'
            for extra_ind in range(5 - len(list(mc_all_count_dict.keys()))):
                mc_all_count_dict[f'do nothing {extra_ind}'] = 0

        # Extract each ensemble
        for ensemble_ind in range(cfg.num_ensemble):

            # Extract
            mc_post_response_data = mc_post_response_data_all[
                data_ind * cfg.num_ensemble + ensemble_ind]
            token = mc_post_response_data.strip().upper(
            )  # assume to be 'A', 'B', 'C', 'D', or 'E'
            assert token in cfg.mc_sigs
            # if token == 'E':
            #     mc_all_count_dict['none'] += 1
            # else:
            token_index = ord(token) - ord('A')
            chosen_mc = data['mc_post'][f'ens_{ensemble_ind}']['mc_all'][
                token_index]
            mc_all_count_dict[chosen_mc] += 1

        # Compute prob
        top_tokens = list(cfg.mc_sigs)
        # top_smx = [
        #     cnt / cfg.num_ensemble for _, cnt in mc_all_count_dict.items()
        # ]
        top_smx = [
            cnt / cfg.num_ensemble for _, cnt in mc_all_count_dict.items()
        ]

        # Print probe prompt
        logging.info(f"========= Probe {data_ind+1}/{cfg.num_data} ========")
        logging.info(data['init']['request'])
        for mc_ind, mc in enumerate(mc_all):
            logging.info(f"{top_tokens[mc_ind]}) {mc}")

        # Determine true label
        true_labels = []
        true_mc_all = []
        none_index = None
        # mc_all = data[f'step_{cur_step}']['mc_all']
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
        # logging.info(f"True label: {true_label}")
        logging.info(f'Top smx: {top_smx}')
        logging.info("===============================\n")

        # Determine action
        # if cfg.calibration_mode:  # choose the label from the true label with the highest logprob - for calibration, we do not care if the top logprob is a true label
        #     if true_labels == [none_sig]:
        #         obj_true, loc_true = random.choice(get_true_action(data))
        #     else:
        #         true_label_inds = [
        #             top_tokens.index(label) for label in true_labels
        #         ]
        #         true_label_logprobs = [
        #             top_logprobs[ind] for ind in true_label_inds
        #         ]
        #         best_label = true_labels[np.argmax(true_label_logprobs)]
        #         true_action = mc_all[top_tokens.index(best_label)]

        #         # extract the object and location from e.g., "get_obj_pos('green triangle') and place at get_pos(l2)"
        #         obj_true = true_action.split(' in')[0].split('put ')[1]
        #         loc_true = ' '.join(true_action.split(' ')[-2:])
        #         if 'plate' not in loc_true:
        #             breakpoint()
        # else:
        #     raise NotImplementedError

        data[f'step_1'] = data_1[f'step_1']
        data[f'step_2'] = data_2[f'step_2']
        # breakpoint()

        # Save
        # data['action'].append([obj_true, loc_true])
        data[f'step_{cur_step}']['top_tokens'] = top_tokens
        data[f'step_{cur_step}']['top_logprobs'] = None
        data[f'step_{cur_step}']['top_smx'] = top_smx
        data[f'step_{cur_step}']['true_label'] = true_labels
        data[f'step_{cur_step}']['add_mc_prefix'] = none_index
        data['top_tokens'] = top_tokens
        data['top_smx'] = top_smx
        data['top_logprobs'] = None
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
    logging.info(f'Number of questions labeled as none: {num_none_label}')
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
