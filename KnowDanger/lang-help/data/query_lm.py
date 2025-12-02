""" 
Query language model with config file.

Load the prompts from a text file with delimiter '--0000--'.

"""

import os
import argparse
import pickle
import logging
import random
from omegaconf import OmegaConf
import openai
from tqdm import tqdm

from agent.language_model import LanguageModel


def main(cfg):

    # LM agent
    lm_agent = LanguageModel(cfg.openai_api_key)

    # Load prompts
    txt_path = os.path.join(cfg.parent_data_folder, cfg.txt_path_from_parent)
    with open(txt_path, 'r') as f:
        prompts = f.read().split('--0000--')

    # Generate data
    response_data_all = []
    response_all = []
    for prompt in tqdm(prompts):
        response_full, response = lm_agent.prompt_gpt_complete(
            prompt,
            lm_model=cfg.lm_model,
            temperature=cfg.lm_temperature,
            max_tokens=cfg.lm_max_token,
            logprobs=cfg.lm_logprobs,
            timeout_seconds=cfg.lm_timeout_seconds,
            logit_bias=cfg.lm_include_token_id_bias
            if cfg.lm_include_token_id_bias else {},
            stop_seq=cfg.lm_stop_seq,
        )
        response_all.append(response)

        # Log
        logging.warning(response_full)

        # Save
        response_data = {
            'prompt': prompt,
            'response': response,
            'response_full': response_full,
        }
        response_data_all.append(response_data)

    # Save
    with open(cfg.data_save_path, 'wb') as f:
        pickle.dump(response_data_all, f)

    # Save text file
    with open(cfg.txt_save_path, 'w') as f:
        f.write('--0000--'.join(response_all))

    # Summary
    logging.warning('\n============== Summary ==============')
    logging.warning(f'Data saved to: {cfg.data_save_path}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    cfg.data_folder = os.path.dirname(args.cfg_file)
    cfg.parent_data_folder = os.path.dirname(cfg.data_folder)

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
    )  # overwrite
    openai.util.logging.getLogger().setLevel(
        logging.WARNING
    )  # suppress openai logging, but have to use warning for other logging

    # Save path
    cfg.data_save_path = os.path.join(
        cfg.data_folder, cfg.data_save_name + '.pkl'
    )
    cfg.txt_save_path = os.path.join(
        cfg.data_folder, cfg.data_save_name + '.txt'
    )

    # run
    random.seed(cfg.seed)
    main(cfg)
