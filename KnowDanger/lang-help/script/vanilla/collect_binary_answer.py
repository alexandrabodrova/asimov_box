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
        cfg.parent_data_folder, cfg.post_data_path_from_parent
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
    for data_ind, (data, post_response_data) in enumerate(
        zip(data_all, post_response_data_all)
    ):

        # Extract
        if cfg.use_palm:
            chosen_token = post_response_data.strip()
            if chosen_token == 'False':
                top_logprobs = [0.0, 1.0]
            else:
                top_logprobs = [1.0, 0.0]
            # top_logprobs = post_response_data.strip().split(' ')
            # assert len(top_logprobs) == 2
            # top_logprobs = [float(logprob) for logprob in top_logprobs]
            top_tokens = ['True', 'False']
        else:
            top_logprobs_full = post_response_data['response_full']["choices"][
                0]["logprobs"]["top_logprobs"][0]
            top_tokens = [token.strip() for token in top_logprobs_full.keys()]
            top_logprobs = [value for value in top_logprobs_full.values()]

        # Print probe prompt
        logging.info(
            f"========== Probe {data_ind+1}/{cfg.num_data} =========="
        )
        logging.info(f"Task: {data['init']['request']}")
        logging.info(f"Action: {data['post']['pre_response']}")
        logging.info(
            'Ground truth: {}'.format(data['init']['request_unambiguous'])
        )
        logging.info("=======")

        # Prompt human to label
        if cfg.dummy_label:
            true_label = [random.choice(['A', 'B', 'C', 'D'])]
        else:
            ground_truth = data['init']['request_unambiguous']
            # I think I know the correction action. I will put the blue bowl behind blue block.
            mc = data['post']['pre_response'].split('I will '
                                                   )[-1][:-1]  # remove period
            ambiguity_name = data['init']['ambiguity_name']
            true_label = check_true_label(ground_truth, mc, ambiguity_name)

        logging.info(f"True label is {true_label}")
        logging.info(f"Top logprobs are {top_logprobs}")
        logging.info("=======")

        # Save
        data['top_tokens'] = top_tokens
        data['top_logprobs'] = top_logprobs
        data['true_label'] = true_label  # either True or False, not a list

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
