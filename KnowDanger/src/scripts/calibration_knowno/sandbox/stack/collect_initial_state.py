""" Multi-step-multi-label, human clarification, partially observable, tabletop manipulation environment

Generate initial states of the stacks in the multi-step setting in the tabletop manipulation environment. Also generate the prompt for the first step.

Partial observability:
    - To introduce ambiguity beyond the first step, use partial observability due to the stack. To ensure reliable detection of the top object of the stack, make sure the object fully occludes the objects below it (by making the objects below it smaller).
    - When the robot is building a new stack to solve the task, naturally the object at the bottom is larger, causing issue with detection. But since we do not need detection for the new stacks (robot knows what it did before, and if we assume action always successfully executed, the robot knows objects and the orders in the new stack), we can just ignore the new stacks in the detection (assuming the new stacks are at the other side of the table).

Actions:
    - We want to define ground truth action(s) for each step. There could be multi-label, e.g., "Sort the blocks by color at L1 and L2", and it is okay to put the first block at either L1 or L2. But sometimes if there is a distrctor, e.g., a circle, we want the ground truth action to be what human specifies. Human can also say "either L1 or L2 is fine", but we avoid this for now.
    - However, it is difficult to define the action sequence here because the true action in the following steps may depend on the action in the previous step. Thus, we define it step-by-step when collecting mc_post data after LM generates multiple choices.

Possible tasks:
1. Sort {object} by color in two new stacks - ambiguity: third color, or a different type of object
2. Sort the objects/items/pieces by color in two new stacks - ambiguity: a third color
3. Sort the objects/items/pieces by type in two new stacks - ambiguity: a third type
4. Move the {color} {object} to a new stack - ambiguity - a new color or object type (the robot may know what to do, e.g., make a new stack)

"""
import os
import argparse
import random
import numpy as np
import pickle
import logging
from omegaconf import OmegaConf


def main(cfg):

    # Possible objects, colors, and tasks
    possible_objects = cfg.possible_objects
    possible_colors = cfg.possible_colors
    possible_place_locations = cfg.possible_place_locations
    possible_task_template_dict = {
        'type_1': 'Sort the {object} by color at L1 and L2.',
        'type_2': 'Sort the objects by color at L1 and L2.',
        'type_3': 'Sort the objects by type at L1 and L2.',
        # 'type_4': 'Move the {color} {object} to a new stack.',
    }
    task_template_ratio = np.array(cfg.task_template_ratio
                                  ) / sum(cfg.task_template_ratio)

    # Sample all init data
    init_data_all = []
    for _ in range(cfg.num_data):

        # Sample task template
        task_template_key = random.choices(
            list(possible_task_template_dict.keys()),
            weights=task_template_ratio
        )[0]
        # task_template_key = 'type_3'
        task_template = possible_task_template_dict[task_template_key]

        # make it three steps - assume only one step is ambiguous
        num_step = 3
        amb_step = -1  # initialize

        # Sample possible states based on the type
        if 'type_1' in task_template_key:
            # 'Sort the {object} by color in two new stacks',

            # Three possible cases: no ambiguity, ambiguity with a third color, ambiguity with a different object type
            amb_type = random.choices([
                'no_amb',
                'third_color',
                'second_type',
            ], weights=cfg.type_1.amb_ratio, k=1)[0]

            # Sample true object, different colors, ambiguous object, and ambiguous color
            if amb_type == 'no_amb':

                # same object
                true_object = random.choice(possible_objects)
                object_steps = [true_object] * num_step

                # two colors
                colors = random.sample(possible_colors, 2)
                while 1:  # make sure the two colors both used
                    color_steps = random.choices(colors, k=num_step)
                    if set(color_steps) == set(colors):
                        break

            elif amb_type == 'third_color':

                # sample object
                true_object = random.choice(possible_objects)
                object_steps = [true_object] * num_step

                # always ambiguous in the last step
                amb_step = 2

                # get three colors and make the last one ambiguous color
                color_steps = random.sample(possible_colors, k=num_step)
                amb_color = color_steps[-1]

            elif amb_type == 'second_type':

                # sample object and ambiguous ones
                true_object, amb_object = random.sample(possible_objects, 2)
                amb_step = random.randint(
                    0, num_step - 1
                )  # first step can be unambiguous
                object_steps = []
                for i in range(num_step):
                    if i == amb_step:
                        object_steps.append(amb_object)
                    else:
                        object_steps.append(true_object)

                # two colors
                colors = random.sample(possible_colors, 2)
                while 1:  # make sure the two colors both used
                    color_steps = random.choices(colors, k=num_step)
                    if set(color_steps) == set(colors):
                        break

            # Fill in template - make it plural
            request = task_template.format(object=true_object + 's')

        elif 'type_2' in task_template_key:
            # "'Sort the objects by color in two new stacks',"

            # Two possible cases: no ambiguity, ambiguity with a third color
            amb_type = random.choices([
                'no_amb',
                'third_color',
            ], weights=cfg.type_2.amb_ratio, k=1)[0]

            # Sample different object
            true_object = random.choices(possible_objects, k=num_step)
            object_steps = true_object

            # Sample colors
            if amb_type == 'no_amb':
                # two colors
                colors = random.sample(possible_colors, 2)
                while 1:  # make sure the two colors both used
                    color_steps = random.choices(colors, k=num_step)
                    if set(color_steps) == set(colors):
                        break
            elif amb_type == 'third_color':
                # always ambiguous in the last step
                amb_step = 2

                # get three colors and make the last one ambiguous color
                color_steps = random.sample(possible_colors, k=num_step)
                amb_color = color_steps[-1]

            # Fill in template
            request = task_template

        elif 'type_3' in task_template_key:
            # 'Sort the objects by type in two new stacks',

            # Two possible cases: no ambiguity, ambiguity with a third type
            amb_type = random.choices([
                'no_amb',
                'third_type',
            ], weights=cfg.type_3.amb_ratio, k=1)[0]

            # Sample colors
            color_steps = random.choices(possible_colors, k=num_step)

            # Sample objects
            if amb_type == 'no_amb':
                # two objects
                objects = random.sample(possible_objects, 2)
                while 1:
                    object_steps = random.choices(objects, k=num_step)
                    if set(object_steps) == set(objects):
                        break
            elif amb_type == 'third_type':
                # always ambiguous in the last step
                amb_step = 2

                # get three objects and make the last one ambiguous object
                object_steps = random.sample(possible_objects, k=num_step)
                amb_object = object_steps[-1]

            # Fill in template
            request = task_template

        elif 'type_4' in task_template_key:
            pass

        # Determine stack orders - assume one stack
        stacks = []
        stack = []  # front means top of the stack
        for i in range(num_step):
            stack.append(f'{color_steps[i]} {object_steps[i]}')
        stacks.append(stack)

        # Log
        logging.info(
            '=============== Data {} ==============='.
            format(len(init_data_all) + 1)
        )
        logging.info('Task: {}'.format(request))
        # logging.info('True object: {}'.format(true_object))
        # logging.info('True colors: {}'.format(colors))
        logging.info('Ambiguous type: {}'.format(amb_type))
        logging.info('Stacks: {}'.format(stacks))
        logging.info('================= END =================\n\n')

        # Save data
        data = {
            'request': request,
            'ambiguity_type': amb_type,
            'ambiguous_step': amb_step,
            #
            'stack': stack,  # assume one stack
            'action': [
            ],  # initialize action executed - e.g., action.append(['blue block', 'L3']) for placing blue block on L3
            # 'true_label_seq': [],
        }
        init_data_all.append(data)

    # Save all data
    with open(cfg.data_save_path, 'wb') as f:
        pickle.dump(init_data_all, f)

    # Print summary
    print("=====================================")
    print('Number of data: {}'.format(len(init_data_all)))
    print(
        'Ambiguity type split: {}'.format([
            len([d
                 for d in init_data_all
                 if d['ambiguity_type'] == amb_type])
            for amb_type in
            ['no_amb', 'third_color', 'third_type', 'second_type']
        ])
    )
    # check duplicate stack
    stack_all = [''.join(d['stack']) for d in init_data_all]
    print(
        'Number of duplicate stacks: {}'.
        format(len(init_data_all) - len(set(stack_all)))
    )
    print("=====================================\n")


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

    # Merge with prompt cfg
    cfg_prompt = OmegaConf.load(
        os.path.join(cfg.parent_data_folder, 'prompt.yaml')
    )
    cfg = OmegaConf.merge(cfg_prompt, cfg)

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
    )

    # Save path
    cfg.data_save_path = os.path.join(
        cfg.data_folder, cfg.data_save_name + '.pkl'
    )

    # Run
    random.seed(cfg.seed)
    main(cfg)
