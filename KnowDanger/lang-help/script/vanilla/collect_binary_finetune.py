""" Single-step, human clarification, tabletop manipulation environment

Combine data with binary ambiguous label for fine-tuning.

Save data in the JSON format of:
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}

From OpenAI documentation:
    For fine-tuning, each training example generally consists of a single input example and its associated output, without the need to give detailed instructions or include multiple examples in the same prompt.

openai tools fine_tunes.prepare_data -f <LOCAL_FILE>

openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> -m <BASE_MODEL>
openai api fine_tunes.follow -i <YOUR_FINE_TUNE_JOB_ID> # resume

# List all created fine-tunes
openai api fine_tunes.list

# Retrieve the state of a fine-tune. The resulting object includes
# job status (which can be one of pending, running, succeeded, or failed)
# and other information
openai api fine_tunes.get -i <YOUR_FINE_TUNE_JOB_ID>

# Cancel a job
openai api fine_tunes.cancel -i <YOUR_FINE_TUNE_JOB_ID>

"""
import os
import argparse
import pickle
import logging
import json
from omegaconf import OmegaConf
from util.vanilla import check_true_label


def main(cfg):

    # Load prev data
    prev_data_path = os.path.join(
        cfg.parent_data_folder, cfg.prev_data_path_from_parent
    )
    with open(prev_data_path, 'rb') as f:
        prev_data_all = pickle.load(f)

    # Initialize json data
    json_data = []

    # Generate data
    for data_ind in range(cfg.num_data):
        data = prev_data_all[data_ind]

        # Determine label
        ground_truth = data['init']['request_unambiguous']
        # I think I know the correction action. I will put the blue bowl behind blue block.
        mc = data['post']['pre_response'].split('I will '
                                               )[-1][:-1]  # remove period
        ambiguity_name = data['init']['ambiguity_name']
        true_label = check_true_label(
            ground_truth, mc, ambiguity_name
        )  # True of False

        # Save json
        json_data.append({
            'prompt': data['post']['post_prompt'],
            'completion': true_label,
        })

    # Save json
    with open(cfg.json_save_path, 'w') as f:
        json.dump(json_data, f)


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

    # Pickle save path
    cfg.data_save_path = os.path.join(
        cfg.data_folder, cfg.data_save_name + '.pkl'
    )

    # JSON save path
    cfg.json_save_path = os.path.join(
        cfg.data_folder, cfg.data_save_name + '.json'
    )

    # Run
    main(cfg)