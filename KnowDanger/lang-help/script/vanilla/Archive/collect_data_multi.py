""" Collect data in the multi-step setting.

Start with the numeric setting only. E.g., "put three blocks left of the yellow bowl."

For now, do not save additional prompt between steps, such as "which of the following options can the next step?"

Allow sem type question, but only for one step.

For each step, still assume there is only one correct answer.

"""

import os
import argparse
from omegaconf import OmegaConf
from string import Template
import pickle
import logging
import random
import seaborn as sns


sns.set(font_scale=2.5)
sns.set_style('darkgrid', {'axes.linewidth': 3, 'axes.edgecolor': 'black'})

from ambiguity import NumericMulti, ATTRIBUTE_AMBIGUITY, SPATIAL_AMBIGUITY, NUMERIC_AMBIGUITY
from util.data import determine_true_label_type, postprocess_mc


MC_LABELS = ['a', 'b', 'c', 'd']
STEP_STR = ['first', 'second', 'third']


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
    num_step = cfg.num_step
    step_prompt_prefix = cfg.step_prompt_prefix
    assert num_step == len(step_prompt_prefix)

    # Class for generating each type of ambiguity
    # attribute_factory = Attribute(cfg, adj_choices, obj_choices, rel_choices, action_choices, mc_template_choices)
    # spatial_factory = Spatial(cfg, adj_choices, obj_choices, rel_choices, action_choices, mc_template_choices)
    numeric_factory = NumericMulti(
        cfg, num_choices, adj_choices, obj_choices, rel_choices,
        action_choices, mc_template_choices
    )

    # Generate data
    data = []
    data_ind = 0
    num_sem_target = int(cfg.num_data * cfg.sem_ratio)
    num_sem = 0
    while data_ind < cfg.num_data:

        # Sample ambiguity type with weights
        ambiguity_type = random.choices([
            ATTRIBUTE_AMBIGUITY, SPATIAL_AMBIGUITY, NUMERIC_AMBIGUITY
        ], weights=cfg.ambiguity_ratio, k=1)[0]
        if ambiguity_type == ATTRIBUTE_AMBIGUITY:
            raise 'Attribute ambiguity not supported yet in multi-step setting!'
        elif ambiguity_type == SPATIAL_AMBIGUITY:
            raise 'Spatial ambiguity not supported yet in multi-step setting!'
        elif ambiguity_type == NUMERIC_AMBIGUITY:
            factory = numeric_factory
        else:
            raise 'Unknown ambiguity type!'
        logging.info(f'# Ambiguity type: {ambiguity_type}')

        # Sample request
        request_template = random.choice(list(request_template_choices))

        # Sample request setting (objects, adjectives, and relations) based on the ambiguity type
        action, obj1, adj1, rel, obj2, adj2, _ = factory.sample_request_setting(
        )

        # Determine if sampling more sem data
        exclude_sem = (num_sem == num_sem_target)

        # Generate context and multiple choices according to the ambiguity type
        request = factory.generate_request(
            request_template, action, obj1, adj1, rel, obj2, adj2
        )
        gen_mc_steps = factory.generate_mc_steps(
            obj1, adj1, rel, obj2, adj2, exclude_sem
        )

        # Re-sample if generate_mc() returns None
        if gen_mc_steps is None:
            continue
        else:
            mc_all_steps, mc_types_steps, info_mc_steps = gen_mc_steps
            data_ind += 1
        logging.info(f'# {data_ind}: {request}')

        # Post-process the sampled multiple choices and also generate labels
        context_steps = []
        true_label_steps = []
        for step, mc_all in enumerate(mc_all_steps):
            logging.info('Step %d:' % (step+1))

            # Prepend context
            context_step = step_prompt_prefix[step].replace(
                '{request}', request
            ).replace('{Step}', STEP_STR[step].capitalize())

            # Post-process multiple choices - add multiple choices to context
            mc_types = mc_types_steps[step]
            mc_all, mc_types, context_step = postprocess_mc(
                context_step, mc_all, mc_types, MC_LABELS,
                add_none_option=False, verbose=True
            )

            # Add answer prompt
            answer_prompt_step = cfg.answer_prompt.replace(
                '{step}', STEP_STR[step]
            )
            context_step += answer_prompt_step

            # Generate labels automatically based on request type
            if ambiguity_type == SPATIAL_AMBIGUITY:
                true_label_steps.append(cfg.e_sig)
            else:
                true_label_step, _ = determine_true_label_type(
                    mc_types, MC_LABELS, use_e_for_multiple_amb=False
                )
                true_label_steps.append(true_label_step)

            # Summarize previous action
            for prev_step in range(step):
                true_label_prev_step = MC_LABELS.index(
                    true_label_steps[prev_step]
                )
                previous_action = mc_all_steps[prev_step][
                    true_label_prev_step].replace('do', 'did')
                context_step = context_step.replace(
                    '{step_' + str(prev_step + 1) + '}', previous_action
                )

            # Save data
            context_steps.append(context_step)
            mc_all_steps[step] = mc_all
            mc_types_steps[step] = mc_types

        # Save data
        data.append({
            'request': request,
            'context_steps': context_steps,
            'mc_steps': mc_all_steps,
            'mc_types_steps': mc_types_steps,
            'request_type': ambiguity_type
        })

        # Determine request type - look at each step
        true_type_steps = []
        for step in range(num_step):
            _, true_type_step = determine_true_label_type(
                mc_types_steps[step], MC_LABELS
            )
            true_type_steps.append(true_type_step)
        # if one step is sem, then the whole question is sem; otherwise, if one step is amb, then the whole question is amb; otherwise, the whole question is eq
        if 'sem' in true_type_steps:
            true_type = 'sem'
        elif 'amb' in true_type_steps:
            true_type = 'amb'
        else:
            true_type = 'eq'
        if true_type == 'sem':
            num_sem += 1

        # log
        logging.info(
            f'True label steps: {true_label_steps}. True type: {true_type}.'
        )
        for step in range(num_step):
            logging.info(f'Context at step {step+1}: {context_steps[step]}')
        logging.info('')
        data[-1]['true_label_steps'] = true_label_steps
        data[-1]['true_type'] = true_type

    # Save all data
    with open(cfg.save_data_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Summarize
    logging.info('\n============== Summary ==============')
    logging.info(f'Number of questions generated: {len(data)}')
    logging.info(
        f'Number of questions of eq type: {len([x for x in data if x["true_type"] == "eq"])}'
    )
    logging.info(
        f'Number of questions of amb type: {len([x for x in data if x["true_type"] == "amb"])}'
    )
    logging.info(
        f'Number of questions of sem type: {len([x for x in data if x["true_type"] == "sem"])}'
    )
    logging.info(
        f'Number of repeated questions: {len(data) - len(set([x["request"] for x in data]))}'
    )
    logging.info(
        f'Number of attribute ambiguity questions: {len([x for x in data if x["request_type"] == ATTRIBUTE_AMBIGUITY])}'
    )
    logging.info(
        f'Number of spatial ambiguity questions: {len([x for x in data if x["request_type"] == SPATIAL_AMBIGUITY])}'
    )
    logging.info(
        f'Number of numeric ambiguity questions: {len([x for x in data if x["request_type"] == NUMERIC_AMBIGUITY])}'
    )
    # count = 0
    # for x in data:
    #     if len(set(x['mc'])) < len(x['mc']):
    #         count += 1
    #         # print(x['context'])
    #         # print(x['true_label'])
    #         # print(x['true_type'])
    #         # print()
    # logging.info(f'Number of questions with repeated multiple choices: {count}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file path", default='', type=str
    )
    args = parser.parse_args()
    if args.cfg_file == '':
        print('Using pre-defined parameters!')
        cfg = OmegaConf.create()
        cfg.num_data = 500
        cfg.num_mc_sample = 4  # number of multiple choices for each question, excluding (e) none of the above
        cfg.max_eq_mc = 1  # maximum number of multiple choices with eq combination
        cfg.max_amb_mc = 3  # maximum number of multiple choices with ambiguous combination
        cfg.amb_mc_ratio = [
            0.5, 0.3, 0.2
        ]  # split of probability of the number of ambiguous multiple choices
        assert cfg.max_amb_mc == len(cfg.amb_mc_ratio)
        cfg.data_folder = 'data/test_122022'
        cfg.answer_prompt = 'Answer:'  # or 'Answer (assign logprob for each option): ', or 'Choose from {a,b,c,d,e}:'
        cfg.adj_choices = None
        cfg.obj_choices = None
    else:
        cfg = OmegaConf.load(args.cfg_file)
    cfg.save_data_path = os.path.join(cfg.data_folder, 'request_data.pkl')
    cfg.logging_path = os.path.join(cfg.data_folder, 'collect_data.log')

    # logging
    logging.basicConfig(
        level=logging.INFO, format='%(message)s', handlers=[
            logging.FileHandler(cfg.logging_path, mode='w'),
            logging.StreamHandler()
        ]
    )  # overwrite

    # Seed
    random.seed(cfg.seed)

    # run
    main(cfg)
