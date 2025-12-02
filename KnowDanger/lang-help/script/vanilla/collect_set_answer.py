""" Single-step, human clarification, tabletop manipulation environment

Combine data with LM response to the multiple choices, for conformal prediction.

"""
import os
import argparse
import pickle
import logging
import random
from omegaconf import OmegaConf
from util.vanilla import check_true_label


def main(cfg):

    # Load answer data
    prev_data_path = os.path.join(
        cfg.parent_data_folder, cfg.prev_data_path_from_parent
    )
    with open(prev_data_path, 'rb') as f:
        data_all = pickle.load(f)

    # Load answer response data
    post_response_data_path = os.path.join(
        cfg.parent_data_folder, cfg.post_response_data_path_from_parent
    )
    if cfg.use_palm:
        with open(post_response_data_path, 'r') as f:
            post_response_data_all = f.read().split('--0000--')
    else:
        with open(post_response_data_path, 'rb') as f:
            post_response_data_all = pickle.load(f)

    # Generate data
    answer_data_all = []
    num_none_label = 0
    for data_ind, (data, post_response_data) in enumerate(
        zip(data_all, post_response_data_all)
    ):

        # Extract
        if cfg.use_palm:
            top_set = post_response_data.split('\n')[0].split(',')
            top_set = [x.strip() for x in top_set]
            top_tokens = ['A', 'B', 'C', 'D', 'E']

            for sig in top_set:
                if sig not in top_tokens:
                    breakpoint()
        # else:
        #     top_logprobs_full = mc_post_response_data['response_full'][
        #         "choices"][0]["logprobs"]["top_logprobs"][0]
        #     top_tokens = [token.strip() for token in top_logprobs_full.keys()]
        #     top_logprobs = [value for value in top_logprobs_full.values()]
        #     flag_rerun = False

        if data['init']['template_mc'] is not None:
            mc_all = data['init']['template_mc']['mc_all']
        else:
            mc_all = data['mc_post']['mc_all']
            mc_all_new = []
            for mc in mc_all:
                if mc != '' and mc != ' ':
                    mc_all_new.append(mc)
            mc_all = mc_all_new

        # Print probe prompt
        logging.info(
            f"========== Probe {data_ind+1}/{cfg.num_data} =========="
        )
        logging.info(data['init']['request'])
        logging.info(mc_all)

        # Prompt human to label
        if cfg.dummy_label:
            true_label = [random.choice(['A', 'B', 'C', 'D'])]
        else:
            # try figuring answer automatically
            ground_truth = data['init']['request_unambiguous']
            # first get move objects
            ambiguity_name = data['init']['ambiguity_name']

            true_label = []
            none_index = None
            # go through the multiple choices
            for i, mc in enumerate(mc_all):
                if mc == 'do nothing':
                    continue
                if 'not listed here' in mc:
                    none_index = i
                    continue
                if len(mc) < 5:
                    continue

                true_label_mc = check_true_label(
                    ground_truth, mc, ambiguity_name
                )
                if true_label_mc == 'True':
                    true_label.append(top_tokens[i])

            if len(true_label) == 0:
                true_label.append(top_tokens[none_index])
                num_none_label += 1
        logging.info(f"True label is {true_label}")
        # logging.info(f"Top logprobs are {top_logprobs}")
        logging.info("=======")

        # while 1:
        #     try:
        #         logging.info(
        #             f"And the ground truth is {data['init']['request_unambiguous']}"
        #         )
        #         true_label = input(
        #             "Please provide label(s) in the format of 'label_1, label_2, ...'; E for none of the above: "
        #         ).split(',')
        #         true_label = [x.strip().upper() for x in true_label]
        #         if len(true_label) < 1:
        #             raise ValueError
        #     except:
        #         continue
        #     break

        # Save
        data['top_tokens'] = top_tokens
        data['top_set'] = top_set
        data['true_label'] = true_label
        # data['flag_rerun'] = flag_rerun
        data['add_mc_prefix'] = top_tokens[none_index]

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
    cfg.parent_data_folder = os.path.dirname(cfg.data_folder)

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