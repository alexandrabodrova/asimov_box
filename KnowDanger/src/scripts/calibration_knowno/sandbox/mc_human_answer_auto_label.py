""" Language model multiple choice prediction with human answer

Collect dataset of LLM-Human interactions with LLM generating multiple choice predictions. Label is automatically generated.

If the LLM is uncertain, human provides answer. No LLM asking clarification questions or human anwering them.

"""

import os
import argparse
from omegaconf import OmegaConf
from string import Template
import pickle
import logging
import random
import openai

from ambiguity import Attribute, Spatial, Numeric, ATTRIBUTE_AMBIGUITY, SPATIAL_AMBIGUITY, NUMERIC_AMBIGUITY
from agent.language_model import prompt_gpt_complete


def main(cfg):
    adj_choices = cfg.adj_choices
    obj_choices = cfg.obj_choices
    num_choices = cfg.numeric_choices
    rel_choices = cfg.rel_choices
    action_choices = cfg.action_choices
    request_template_choices = [
        Template('$action the $adj1 $obj1 $rel the $adj2 $obj2'),
    ]
    mc_template_choices = [
        Template('$action the $adj1 $obj1 $rel_phrase'),
    ]
    question_prompt = cfg.question_prompt

    # Class for generating each type of ambiguity
    attribute_factory = Attribute(
        cfg, adj_choices, obj_choices, rel_choices, action_choices,
        mc_template_choices
    )
    spatial_factory = Spatial(
        cfg, adj_choices, obj_choices, rel_choices, action_choices,
        mc_template_choices
    )
    numeric_factory = Numeric(
        cfg, num_choices, adj_choices, obj_choices, rel_choices,
        action_choices, mc_template_choices
    )

    # Load LM data if available, for debugging purposes and saving money - otherwise, save LM data in this run
    if cfg.lm_response_load_path != '':
        with open(cfg.lm_response_load_path, 'rb') as handle:
            lm_response_data = pickle.load(handle)
        logging.warning(
            f'Loaded LM response data: {cfg.lm_response_load_path}'
        )
        flag_load_lm_response_data = True
    else:
        lm_response_data = []
        flag_load_lm_response_data = False

    # Generate data
    data = []
    data_ind = 0
    num_sem = 0
    while data_ind < cfg.num_data:

        # Sample ambiguity type with weights
        ambiguity_type = random.choices([
            ATTRIBUTE_AMBIGUITY, SPATIAL_AMBIGUITY, NUMERIC_AMBIGUITY
        ], weights=cfg.ambiguity_ratio, k=1)[0]
        if ambiguity_type == ATTRIBUTE_AMBIGUITY:
            factory = attribute_factory
        elif ambiguity_type == SPATIAL_AMBIGUITY:
            factory = spatial_factory
        elif ambiguity_type == NUMERIC_AMBIGUITY:
            factory = numeric_factory
        else:
            raise 'Unknown ambiguity type!'

        # Sample request
        request_template = random.choice(list(request_template_choices))

        # Sample request setting (objects, adjectives, and relations) based on the ambiguity type
        action, obj1, adj1, rel, obj2, adj2, _ = factory.sample_request_setting(
        )

        # Generate context and multiple choices according to the ambiguity type
        request = factory.generate_request(
            request_template, action, obj1, adj1, rel, obj2, adj2
        )

        # Sample a ground truth request - ignore action right now
        obj1_true, adj1_true, rel_true, obj2_true, adj2_true = factory.sample_ground_truth_words(
            obj1, adj1, rel, obj2, adj2
        )

        # remove space before comma in the request string
        request = request.replace(' ,', ',')
        logging.warning('----------------------------------------')
        logging.warning(
            f'{data_ind+1}: {request} - ambiguity type: {ambiguity_type}'
        )

        # Get full prompt
        full_mc_prompt = cfg.mc_prompt_prefix.replace("{request}", request)
        # logging.warning('\n------')
        # logging.warning('MC prompt:')
        # logging.warning(full_mc_prompt)
        # logging.warning('------\n')

        # Prompt LM for multiple choices
        if not flag_load_lm_response_data:
            mc_response_full, mc_response = prompt_gpt_complete(
                full_mc_prompt,
                api_key=cfg.gpt_api_key,
                lm_model=cfg.lm_model_mc,
                temperature=cfg.lm_temperature,
                # stop=['#', 'objects ='],
                max_tokens=cfg.mc_max_token
            )
        else:
            mc_response_full, mc_response = lm_response_data[data_ind][:2]

        # Flag for not using this data
        flag_quit = False

        # Process generated mc - shuffle
        mc_all = mc_response.split('\n')
        try:
            for i in range(len(mc_all)):
                mc_all[i] = mc_all[i][2:]  # remove 1., 2., ...
                mc_all[i] = mc_all[i].strip().lower()
                # remove dot at the end if there is one
                if mc_all[i][-1] == '.':
                    mc_all[i] = mc_all[i][:-1]
        except:
            logging.warning('Weird mc:')
            logging.warning(mc_response)
            continue
        random.shuffle(mc_all)
        mc_types = []
        prefix_all = ['a) ', 'b) ', 'c) ', 'd) ']
        mc_prompt_for_answer = ''
        for prefix, mc in zip(prefix_all, mc_all):

            # Sometimes LM generates weird mc that fail to parse
            if len(mc) == 0:
                logging.warning('Weird mc')
                logging.warning(mc_response)
                flag_quit = True
                break

            # Add prefix
            mc = prefix + mc
            mc_prompt_for_answer += mc + '\n'

            # Extract adj and obj pairs, and relation
            mc_words = mc.split(' ')
            adj_obj_pairs = []
            mc_rel_candidate_all = [
            ]  # find all, and then find the highest priority one
            for ind, word in enumerate(mc_words):
                if word in obj_choices.keys():
                    # if word in ['block', 'bowl']:
                    adj = mc_words[ind - 1]
                    obj = mc_words[ind]
                    adj_obj_pairs.append((adj, obj))

                # check if the mc contains the relation
                for key, value in rel_choices.items():
                    if word in value.keywords:
                        mc_rel_candidate_all.append((key, value.priority))

            # highest priority, if multiple with same highest priority, report error
            mc_rel_candidate_all = sorted(
                mc_rel_candidate_all, key=lambda x: x[1]
            )
            if len(mc_rel_candidate_all) > 0:
                if len(mc_rel_candidate_all) > 1 and mc_rel_candidate_all[0][
                    1] == mc_rel_candidate_all[1][1]:
                    logging.warning('Multiple relations with same priority')
                    logging.warning(mc_response)
                    flag_quit = True
                    break
                mc_rel = mc_rel_candidate_all[0][0]
            else:
                mc_rel = None
            # logging.warning('Found adj-obj pairs and rel: {} {}'.format(adj_obj_pairs, mc_rel))

            # Determine the type of mc
            if len(
                adj_obj_pairs
            ) < 2:  # only one object in mc, definitely does not work
                mc_types.append('sem')
            elif mc_rel is None:  # relation cannot be parsed
                mc_types.append('sem')
            elif adj1_true is None:  # request cannot be realized
                mc_types.append('sem')
            else:
                # check if mc has ground truth words
                mc_adj1, mc_obj1 = adj_obj_pairs[0]  # first pair
                mc_adj2, mc_obj2 = adj_obj_pairs[-1]  # last pair
                if mc_adj1 in adj_choices[adj1_true]['eq'] and \
                   mc_obj1 in obj_choices[obj1_true]['eq'] and \
                   mc_adj2 in adj_choices[adj2_true]['eq'] and \
                   mc_obj2 in obj_choices[obj2_true]['eq'] and \
                   mc_rel in rel_choices[rel_true]['eq']:  # all eq
                    mc_types.append('eq')
                elif mc_adj1 in adj_choices[adj1_true]['sem'] or \
                     mc_obj1 in obj_choices[obj1_true]['sem'] or \
                     mc_adj2 in adj_choices[adj2_true]['sem'] or \
                     mc_obj2 in obj_choices[obj2_true]['sem'] or \
                     mc_rel in rel_choices[rel_true]['sem']:   # any sem
                    mc_types.append('sem')
                elif (mc_adj1 in adj_choices[adj1_true]['amb'] or mc_adj1 in adj_choices[adj1_true]['eq']) and \
                    (mc_obj1 in obj_choices[obj1_true]['amb'] or mc_obj1 in obj_choices[obj1_true]['eq']) and \
                    (mc_adj2 in adj_choices[adj2_true]['amb'] or mc_adj2 in adj_choices[adj2_true]['eq']) and \
                    (mc_obj2 in obj_choices[obj2_true]['amb'] or mc_obj2 in obj_choices[obj2_true]['eq']) and \
                    (mc_rel in rel_choices[rel_true]['amb'] or mc_rel in rel_choices[rel_true]['eq']):  # all amb oe eq
                    mc_types.append('amb')
                else:
                    mc_types.append('sem')
        if flag_quit:
            continue

        # Label true answer
        true_label_conversion = {
            0: 'a',
            1: 'b',
            2: 'c',
            3: 'd',
            4: 'e'
        }  # TODO: add to cfg
        num_eq = mc_types.count('eq')
        if num_eq > 1:
            logging.warning('Multiple eq in multiple choices! Skip.')
            continue
        if 'eq' in mc_types:
            true_type = 'eq'
            true_label = true_label_conversion[mc_types.index('eq')]
        elif 'amb' in mc_types:  # 'none of the options' is amb
            true_type = 'amb'
            amb_ind_all = [i for i, x in enumerate(mc_types) if x == 'amb']
            true_label = true_label_conversion[random.choice(amb_ind_all)]
        else:
            true_type = 'sem'
            true_label = 'e'
            num_sem += 1

        ########################## Ask LM to choose the correct answer ##########################

        # Append the question prompt
        full_answer_prompt = cfg.answer_prompt_prefix.replace(
            '{request}', request
        ) + mc_prompt_for_answer + question_prompt
        logging.warning(
            f'\nground truth: {adj1_true, obj1_true, rel_true, adj2_true, obj2_true}'
        )
        # logging.warning('\n------')
        # logging.warning('Answer prompt:')
        logging.warning('\nMultiple choices:')
        logging.warning(mc_prompt_for_answer)
        # logging.warning('------\n')

        # Ask LM to choose the answer
        if not flag_load_lm_response_data:
            answer_response_full, answer_response = prompt_gpt_complete(
                full_answer_prompt,
                api_key=cfg.gpt_api_key,
                lm_model=cfg.lm_model_ans,
                temperature=cfg.lm_temperature_ans,
                max_tokens=1,
                logprobs=5,
                timeout_seconds=5,
                logit_bias=cfg.include_token_id_bias,
            )
        else:
            answer_response_full, answer_response = lm_response_data[data_ind][
                -2:]

        # Save data
        data.append({
            'request': request,
            'context':
                cfg.answer_prompt_prefix.replace('{request}', request)
                + mc_prompt_for_answer,
            'full_question_prompt': full_mc_prompt,
            'full_answer_prompt': full_answer_prompt,
            'lm_response': answer_response_full,
            'mc_response_full': mc_response_full,
            'mc': mc_all,
            'mc_types': mc_types,
            'true_label': true_label,
            'true_type': true_type,
        })
        lm_response_data.append([
            mc_response_full,
            mc_response,
            answer_response_full,
            answer_response,
        ])

        # extract top tokens and logprobs
        top_logprobs_full = answer_response_full["choices"][0]["logprobs"][
            "top_logprobs"][0]
        top_token = answer_response_full["choices"][0]["text"].strip()
        top_tokens = [
            token.strip().lower() for token in top_logprobs_full.keys()
        ]
        top_logprobs = [value for value in top_logprobs_full.values()]

        # Debug
        logging.warning(f'MC types: {mc_types}')
        logging.warning(f'Top tokens: {top_tokens}')
        logging.warning(f'logprobs: {top_logprobs}')
        logging.warning(
            f'True label: {true_label}; Ans: {top_token}; True type: {true_type}'
        )
        logging.warning('----------------------------------------')
        logging.warning('')

        # Count
        data_ind += 1

    # Save data immediately to save money...
    if not flag_load_lm_response_data:
        with open(cfg.lm_response_save_path, 'wb') as handle:
            pickle.dump(
                lm_response_data, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    # Save data to pickle file
    with open(cfg.lm_mc_answer_save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Summary
    logging.warning('\n============== Summary ==============')
    logging.warning(f'Number of questions generated: {len(data)}')
    logging.warning(
        f'Number of questions of eq type: {len([x for x in data if x["true_type"] == "eq"])}'
    )
    logging.warning(
        f'Number of questions of amb type: {len([x for x in data if x["true_type"] == "amb"])}'
    )
    logging.warning(
        f'Number of questions of sem type: {len([x for x in data if x["true_type"] == "sem"])}'
    )
    # logging.warning(f'Number of repeated questions: {len(data) - len(set([x["context"] for x in data]))}')
    count = 0
    for x in data:
        if len(set(x['mc'])) < len(x['mc']):
            count += 1
            # print(x['context'])
            # print(x['true_label'])
            # print(x['true_type'])
            # print()
    logging.warning(
        f'Number of questions with repeated multiple choices: {count}\n'
    )
    logging.warning(
        f'Ratio of true label reported as the highest logprob by LM: {sum([x["true_label"] == x["lm_response"]["choices"][0]["text"].strip() for x in data])/cfg.num_data}'
    )
    data_type_split = [
        len([d for d in data if d['true_type'] == 'eq']),
        len([d for d in data if d['true_type'] == 'amb']),
        len([d for d in data if d['true_type'] == 'sem']),
    ]
    logging.warning(f'Number of questions of each type: {data_type_split}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    if args.cfg_file == '':
        print('Using pre-defined parameters!')
        cfg = OmegaConf.create()
        cfg.num_data = 100
        cfg.data_folder = 'data/v2_prompt/collect'
        cfg.prompt_prefix = 'Request: generate four different choices of next action in plain English'
        # cfg.answer_prompt = 'Answer:' # or 'Answer (assign logprob for each option): ', or 'Choose from {a,b,c,d,e}:'
        cfg.adj_choices = None
        cfg.obj_choices = None
        cfg.lm_model = 'text-davinci-003'
        cfg.gpt_temperature = 0.0
    else:
        cfg = OmegaConf.load(args.cfg_file)
    cfg.logging_path = os.path.join(
        cfg.data_folder, cfg.log_file_name + '.log'
    )
    cfg.lm_mc_answer_save_path = os.path.join(
        cfg.data_folder, cfg.lm_mc_answer_save_file_name + '.pkl'
    )
    cfg.lm_response_save_path = os.path.join(
        cfg.data_folder, cfg.lm_response_save_file_name + '.pkl'
    )
    if cfg.lm_response_load_file_name != '':
        cfg.lm_response_load_path = os.path.join(
            cfg.data_folder, cfg.lm_response_load_file_name + '.pkl'
        )
    else:
        cfg.lm_response_load_path = ''

    # logging
    logging.basicConfig(
        level=logging.INFO, format='%(message)s', handlers=[
            logging.FileHandler(cfg.logging_path, mode='w'),
            logging.StreamHandler()
        ]
    )  # overwrite
    openai.util.logging.getLogger().setLevel(
        logging.WARNING
    )  # suppress openai logging, but have to use warning for other logging

    # run
    random.seed(cfg.seed)
    main(cfg)
