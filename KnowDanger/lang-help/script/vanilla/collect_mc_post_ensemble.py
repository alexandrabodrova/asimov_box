""" Single-step, tabletop manipulation environment

Collect dataset of prompting LM to choose from the multiple choices (i.e., exeucting the action). Use ensemble by randomly choosing an example from a set.

TODO: implement for template MC

"""
import os
import argparse
import pickle
import logging
import random
from omegaconf import OmegaConf
from agent.multiple_choice import MultipleChoice
from util.data import postprocess_mc


def main(cfg):

    # Load example set
    with open(
        os.path.join(cfg.parent_data_folder, cfg.example_set_path_from_parent),
        'r'
    ) as f:
        example_set = f.read().split('--0000--')

    # Load background prompt
    with open(
        os.path.join(
            cfg.parent_data_folder, cfg.background_prompt_path_from_parent
        ), 'r'
    ) as f:
        background_prompt = f.read()

    # Load mc_post prompt
    with open(
        os.path.join(
            cfg.parent_data_folder, cfg.mc_post_prompt_path_from_parent
        ), 'r'
    ) as f:
        raw_mc_post_prompt = f.read()

    # Multiple choice agent
    mc_agent = MultipleChoice()

    # Load previous data
    prev_data_path = os.path.join(
        cfg.parent_data_folder, cfg.prev_data_path_from_parent
    )
    with open(prev_data_path, 'rb') as f:
        data_all = pickle.load(f)

    # Load mc response data if specified
    if cfg.mc_template:  # no LLM response then
        mc_pre_response_data_all = [None] * cfg.num_data
    else:
        mc_pre_response_data_path = os.path.join(
            cfg.parent_data_folder, cfg.mc_pre_response_data_path_from_parent
        )
        with open(mc_pre_response_data_path, 'r') as f:
            mc_pre_response_data_all = f.read().split('--0000--')

    # Generate data
    mc_post_data_all = []
    mc_post_prompt_all = []
    flag_prompt_verify = True
    for data_ind, (data, mc_pre_response_data) in enumerate(
        zip(data_all, mc_pre_response_data_all)
    ):

        # Template MC
        if cfg.mc_template:
            mc_pre_response = None
            template_mc_data = data['init']['template_mc']

            # Post-process the sampled multiple choices
            mc_prompt, mc_all, mc_types = postprocess_mc(
                template_mc_data['mc_all'],
                template_mc_data['mc_types'],
                cfg.mc_sigs,
                add_mc=cfg.add_mc,
                verbose=False,
            )

            # Print probe prompt
            logging.info(f"========== {data_ind+1}/{cfg.num_data} ==========")
            logging.info(data['init']['request'])
            logging.info(mc_prompt)

        # Prompted MC
        else:
            mc_pre_response = mc_pre_response_data.strip()

        # Save
        data['mc_post'] = {}
        data['mc_post']['mc_pre_response'] = mc_pre_response
        data['mc_post']['background_prompt'] = background_prompt

        # Sample an example
        for ensemble_ind in range(cfg.num_ensemble):

            # multiple example
            example_prompt = random.sample(
                example_set, cfg.num_example_per_data
            )
            example_prompt = '\n\n'.join(example_prompt)

            # Randomize order of choices, add prefix
            mc_prompt, mc_all, _ = mc_agent.process_multiple_choice(
                mc_pre_response,
                add_mc=cfg.add_mc,
            )

            # Get action prompt
            mc_post_prompt = raw_mc_post_prompt.replace(
                '{background}', background_prompt
            )
            mc_post_prompt = mc_post_prompt.replace(
                '{example}', example_prompt
            )
            mc_post_prompt = mc_post_prompt.replace(
                '{scene_description}', data['init']['scene_description']
            )
            mc_post_prompt = mc_post_prompt.replace(
                '{request}', data['init']['request']
            )
            mc_post_prompt = mc_post_prompt.replace('{mc}', mc_prompt)

            # Verify action prompt once
            if flag_prompt_verify:
                logging.info("===========")
                input(mc_post_prompt + '\n\nPress any key to verify...')
                flag_prompt_verify = False

            # Save
            ensemble_ind_str = 'ens_' + str(ensemble_ind)
            data['mc_post'][ensemble_ind_str] = {}
            data['mc_post'][ensemble_ind_str]['mc_all'] = mc_all
            data['mc_post'][ensemble_ind_str]['mc_prompt'] = mc_prompt
            data['mc_post'][ensemble_ind_str]['mc_post_prompt'
                                             ] = mc_post_prompt
            mc_post_prompt_all.append(mc_post_prompt)

        # Save
        mc_post_data_all.append(data)
        logging.info("========================\n")

    # Save
    with open(cfg.data_save_path, 'wb') as f:
        pickle.dump(mc_post_data_all, f)

    # Save text file
    with open(cfg.txt_save_path, 'w') as f:
        f.write('--0000--'.join(mc_post_prompt_all))

    # Summary
    logging.info('\n============== Summary ==============')
    logging.info(f'Number of questions generated: {cfg.num_data}')
    logging.info(f'Data saved to: {cfg.data_save_path}.')
    logging.info(f'Prompt saved to: {cfg.txt_save_path}.')


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
    cfg.txt_save_path = os.path.join(
        cfg.data_folder, cfg.data_save_name + '.txt'
    )

    # run
    random.seed(cfg.seed)
    main(cfg)