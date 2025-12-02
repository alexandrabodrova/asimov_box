""" Single-step-single-label, human clarification, tabletop bimanual environment

Generate a dataset of bimanual tasks, corresponding correct action, and the scene description. Do not save the prompt yet.

Very diverse distribution.


"""
import os
import logging
import argparse
import pickle
import random
from omegaconf import OmegaConf


def main(cfg):

    # Sampling weights of the tasks
    task_weights = [
        cfg.task_templates[task].weight for task in cfg.task_templates
    ]

    # Collect tasks
    init_data_all = []
    for data_ind in range(cfg.num_data):

        # Sample a task and get the corresponding config
        task = random.choices(
            list(cfg.task_templates.keys()), weights=task_weights
        )[0]
        task_cfg = cfg.task_templates[task]

        # Sample object
        obj = random.choice(task_cfg.object)

        # Sample true action
        if 'object_from_side' in task_cfg:
            if obj in task_cfg.object_from_side:
                true_action = 'right side'
            else:
                true_action = 'left top'
        else:
            true_action = random.choice(task_cfg.true_action)

        # Get additional background
        additional_background = task_cfg.additional_background

        # Fill in the task
        task = task.replace('{object}', obj)

        # Save data
        init_data_all.append({
            'request': task,
            'additional_background': additional_background,
            'object': obj,
            'true_action': true_action,
            'example_file': task_cfg.example_file,
        })

        # Log
        logging.info('============ Data {} ============'.format(data_ind + 1))
        logging.info(f'Task: {task}')
        logging.info(f'Object: {obj}')
        logging.info(f'True action: {true_action}')
        # logging.info(f'Scene description: {scene}')
        # logging.info(f'Ambiguous? {ambiguous}')
        logging.info(
            f'=========== END {data_ind+1}/{cfg.num_data} ===========\n\n'
        )

    # Save all sampled data
    with open(cfg.data_save_path, 'wb') as handle:
        pickle.dump(init_data_all, handle, protocol=pickle.HIGHEST_PROTOCOL)


if '__main__' == __name__:
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

    # Run
    random.seed(cfg.seed)
    main(cfg)