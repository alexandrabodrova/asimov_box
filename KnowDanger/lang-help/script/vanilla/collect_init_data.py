""" Single-step, tabletop manipulation environment

Collect dataset of the initial states (environment and request). For now, do not use infeasible request (e.g., "move a plate to ..." but there is no plate on the table).

"""
import os
import argparse
import pickle
import logging
import random
from omegaconf import OmegaConf
from agent.task import Task


def main(cfg):

    # Task agent
    task_agent = Task(cfg)

    # Generate data
    init_data_all = []
    for data_ind in range(cfg.num_data):

        # Sample request and the ground truth
        request, info = task_agent.sample_request()
        request = request[0].upper() + request[1:]

        # Get multiple choices if using template
        if cfg.mc_template:
            _, obj1, adj1, rel, obj2, adj2 = info.sample_words.values()
            mc_all, mc_types, mc_info = task_agent.generate_mc(
                obj1, adj1, rel, obj2, adj2, info.ambiguity_type
            )
            template_mc = {
                'mc_all': mc_all,
                'mc_types': mc_types,
                'mc_info': mc_info
            }
        else:
            template_mc = None

        # Determine the scene description depending on the ambiguity type.
        scene_description = cfg.scene_description_template
        for ind, obj in enumerate(info.object_set):
            scene_description = scene_description.replace(f'obj{ind+1}', obj)

        # Save
        init_data_all.append({
             'init': {   # make a new key for each step
                'request': request,
                'scene_description': scene_description,
                'ambiguity_name': info.ambiguity_name,
                'request_unambiguous': info.request_unambiguous,
                'template_mc': template_mc,
            }
        })

        # Log
        logging.info(f'-------------- Data # {data_ind} --------------')
        logging.info(f'Request: {request}')
        logging.info(f'Ambiguity name: {info.ambiguity_name}')
        logging.info(f'Ground truth: {info.request_unambiguous}')
        logging.info(f'Scene description: {scene_description}')
        if cfg.mc_template:
            logging.info('Multiple choices:')
            for mc in mc_all:
                logging.info(f'\t{mc}')
        logging.info('---------------------------------------\n')

    # Save
    with open(cfg.data_save_path, 'wb') as f:
        pickle.dump(init_data_all, f)

    # Summary
    task_request_all = [data['init']['request'] for data in init_data_all]
    num_repeated = len(task_request_all) - len(set(task_request_all))
    logging.info('\n============== Summary ==============')
    logging.info(f'Number of questions generated: {cfg.num_data}')
    logging.info(f'Data saved to: {cfg.data_save_path}.')
    logging.info(f'Number of repeated requests: {num_repeated}')


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

    # Run
    random.seed(cfg.seed)
    main(cfg)
