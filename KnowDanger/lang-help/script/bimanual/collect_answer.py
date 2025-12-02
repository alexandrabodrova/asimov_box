""" Single-step-single-label, Bimanual environment

Collect true label from human.

"""
import os
import argparse
import pickle
import logging
import random
import numpy as np
from omegaconf import OmegaConf


def main(cfg):

    # Load prevc data
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
        mc_post_response_data_all = np.loadtxt(
            mc_post_response_data_path, dtype=str, delimiter='\t',
            encoding='utf-8-sig'
        )
    else:
        with open(mc_post_response_data_path, 'rb') as f:
            mc_post_response_data_all = pickle.load(f)

    # Generate data
    answer_data_all = []
    for data_ind, (data, mc_post_response_data) in enumerate(
        zip(data_all, mc_post_response_data_all)
    ):
        if data_ind < cfg.start_ind:
            continue

        # Extract
        if cfg.use_palm:
            logprob_split = mc_post_response_data.split(' ')
            if len(logprob_split) < 5 or any(
                logprob == '' for logprob in logprob_split
            ):
                logging.info('Missing scores - rerun the data')
                flag_rerun = True
                top_logprobs = None
            else:
                flag_rerun = False
                top_logprobs = [float(v) for v in logprob_split]
            top_tokens = ['A', 'B', 'C', 'D', 'E']
        else:
            top_logprobs_full = mc_post_response_data['response_full'][
                "choices"][0]["logprobs"]["top_logprobs"][0]
            top_tokens = [token.strip() for token in top_logprobs_full.keys()]
            top_logprobs = [value for value in top_logprobs_full.values()]

        # Prompt human to label
        if cfg.dummy_label:
            true_label = [random.choice(['A', 'B', 'C', 'D'])]
        else:
            # while 1:
            #     try:
            #         logging.info(
            #             f"And the true action is {data['true_action']}"
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
            true_label = []
            true_action_split = data['true_action'].split(' ')
            for ind, mc in enumerate(data['mc_all']):
                if all(split in mc.lower() for split in true_action_split):
                    true_label.append(cfg.mc_sigs[ind])
            assert len(true_label) <= 1

        # Prompt human to label true action if E is chosen - this should only happen with E as the only label
        # if len(true_label) == 0:
        #     true_label = [data['add_mc_prefix']]
        #     while 1:
        #         try:
        #             true_action = input(
        #                 "Please label the true action in the format of {object}_{action}"
        #             )
        #         except:
        #             continue
        #         break
        # else:
        #     true_action = None  # no need to label if E is not chosen

        logging.info(
            f"============== Probe {data_ind+1}/{cfg.num_data} =============="
        )
        logging.info(f"Task: {data['request']}")
        logging.info(data['mc_prompt'])
        logging.info(f"logprobs: {top_logprobs}")
        logging.info(f"True action: {data['true_action']}")
        logging.info(f"True label: {true_label}")
        logging.info("=======")
        # input()

        # Save
        data['top_tokens'] = top_tokens
        data['top_logprobs'] = top_logprobs
        data['true_label'] = true_label
        # data['true_action'] = true_action
        data['flag_rerun'] = flag_rerun

        # Save
        answer_data_all.append(data)

        # Save single data
        with open(
            os.path.join(cfg.single_data_save_dir, f'{data_ind}.pkl'), 'wb'
        ) as f:
            pickle.dump(data, f)

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
    cfg.single_data_save_dir = os.path.join(cfg.data_folder, 'single_data')

    # run
    random.seed(cfg.seed)
    main(cfg)
