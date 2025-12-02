""" Multi-step, human clarification, partially observable, tabletop manipulation environment

Collect the human clarification for the question asked by the language model.

For now, assume there is only one clarification round, and after clarification, human asks LM to generate the options, but do not ask for answer yet.

"""

import os
import argparse
import pickle
import logging
import random
from omegaconf import OmegaConf


def main(cfg):

    # Load previous prompt data
    with open(cfg.prev_prompt_path, 'rb') as f:
        prev_prompt_all = pickle.load(f)

    # Load LM response
    with open(cfg.lm_response_path, 'r') as f:
        lm_response = f.read()
    lm_response_all = lm_response.split('--0000--')
    assert len(lm_response_all) == len(prev_prompt_all)

    # with open(
    #     'data/v4_clarify_binary_ss/collect/step_1_clarify.pkl', 'rb'
    # ) as f:
    #     new_data_all = pickle.load(f)
    # with open(
    #     'data/v4_clarify_binary_ms/collect/step_1_clarify.pkl', 'rb'
    # ) as f:
    #     new_data_all = pickle.load(f)

    # Generate data
    num_data = len(prev_prompt_all)
    data_all = []
    for data_ind, (prev_prompt, lm_response) in enumerate(
        zip(prev_prompt_all, lm_response_all)
    ):
        print(prev_prompt['prompt'])

        # TODO: remove PaLM's human response...
        lm_response = lm_response.split('Human')[0].strip()

        print(lm_response)

        # # Get top object
        top_obj = prev_prompt['prompt'].split('There is a')[2].split(
            'at the top'
        )[0].strip()
        scene_prompt = cfg.scene_prompt.replace('{top_object}', top_obj)
        # scene_prompt = cfg.scene_prompt

        # Get request from prev_prompt
        request = prev_prompt['prompt'].split('Task:')[2].split('Human:'
                                                               )[0].strip()
        # request = prev_prompt['prompt'].split('Task:')[1].split('Human:'
        #    )[0].strip()
        task_prompt = cfg.task_prompt.replace('{request}', request)

        # Check if clarification is needed
        flag_clarify = False
        if lm_response.split('.')[0] == 'Uncertain':
            flag_clarify = True

        # Prompt human for clarification answer
        # human_response = new_data['human_response']
        # if human_response is not None:
        #     flag_clarify = True
        # else:
        #     flag_clarify = False
        if flag_clarify:
            while 1:
                try:
                    human_response = input('Please help LM:\n')
                except:
                    continue
                break
        else:
            human_response = None

        # Get new prompt for generating options
        if flag_clarify:
            prompt = cfg.background_prompt + '\n\n' \
                + scene_prompt + '\n\n' \
                + task_prompt + '\n\n' \
                + cfg.probe_uncertainty_prompt + ' ' + lm_response + '\n\n' \
                + 'Human: ' + human_response + '\n\n' \
                + cfg.mc_for_action_prompt
        else:
            prompt = cfg.background_prompt + '\n\n' \
                + scene_prompt + '\n\n' \
                + task_prompt + '\n\n' \
                + cfg.probe_uncertainty_prompt + ' ' + lm_response + '\n\n' \
                + cfg.mc_for_action_no_clarify_prompt

        #
        prompt = prompt.replace('Robot:', 'You:').replace('Human:', 'We:')

        # Log
        logging.info(
            '============ Data {}, new prompt ============'.
            format(data_ind + 1)
        )
        logging.info(prompt)
        logging.info(
            f'================= END {data_ind+1}/{num_data} =================\n\n'
        )

        # Save data
        data_all.append({
            'human_response': human_response,
            'lm_probe_response': lm_response,
            'prompt': prompt,
            # 'info': info,
        })

    # Save all data
    with open(cfg.data_save_path, 'wb') as f:
        pickle.dump(data_all, f)

    # Summary
    logging.info('\n============== Summary ==============')
    logging.info(f'Number of questions generated: {len(data_all)}')
    logging.info(f'Data saved to: {cfg.data_save_path}.')
    logging.info('=====================================')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    cfg.logging_path = os.path.join(
        cfg.data_folder, cfg.log_file_name + '.log'
    )
    cfg.data_save_path = os.path.join(
        cfg.data_folder, cfg.data_save_name + '.pkl'
    )

    # logging
    logging.basicConfig(
        level=logging.INFO, format='%(message)s', handlers=[
            logging.FileHandler(cfg.logging_path, mode='w'),
            logging.StreamHandler()
        ]
    )

    # run
    random.seed(cfg.seed)
    main(cfg)
