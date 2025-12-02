""" Single-step-single-label, human clarification, tabletop bimanual environment

Generate a dataset of bimanual tasks, corresponding correct action, and the scene description. Do not save the prompt yet.

Save:
    - task request
    - objects and initial locations
    - true action: (arm, object, location)

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

    # Aggregate all objects
    all_objects = []
    for category in cfg.categories:
        all_objects += cfg.categories[category]
    num_target_obj = cfg.num_target_obj

    # Collect tasks
    init_data_all = []
    for data_ind in range(cfg.num_data):

        # Sample a task and get the corresponding config
        task = random.choices(
            list(cfg.task_templates.keys()), weights=task_weights
        )[0]
        task_cfg = cfg.task_templates[task]

        # Sample objects according to the sampled task, and also sample the true action
        if task == 'Place the {attribute} object on the table.':
            attribute = random.choice(list(cfg.attributes.keys()))
            objects = random.sample(cfg.attributes[attribute].object, 2)
            object_locations = [(objects[0], 'left'), (objects[1], 'right')]
            move_object = random.choice(objects)
            arm = 'left arm' if move_object == objects[0] else 'right arm'
            true_action = (arm, move_object, 'table center')
            task = task.replace('{attribute}', attribute)
            ambiguous = True

        elif task == 'Place the {category} on the table.':
            category = random.choice(list(cfg.categories.keys()))
            objects = random.sample(cfg.categories[category], 2)
            object_locations = [(objects[0], 'left'), (objects[1], 'right')]
            move_object = random.choice(objects)
            arm = 'left arm' if move_object == objects[0] else 'right arm'
            true_action = (arm, move_object, 'table center')
            task = task.replace('{category}', category)
            ambiguous = True

        elif task == 'Move the {attribute} object to the other bin.':
            # need to move the object to the table first, and then the other arm moves it to the bin
            attribute = random.choice(list(cfg.attributes.keys()))
            objects = random.sample(cfg.attributes[attribute].object, 2)
            object_locations = [(objects[0], 'left'), (objects[1], 'right')]
            move_object = random.choice(objects)
            arm = 'left arm' if move_object == objects[0] else 'right arm'
            other_arm = 'left arm' if arm == 'right arm' else 'right arm'
            true_action = (
                arm, move_object, 'table center', other_arm, move_object,
                'left' if move_object == objects[1] else 'right'
            )
            task = task.replace('{attribute}', attribute)
            ambiguous = True

        elif task == 'Move the {category} to the other bin.':
            # need to move the object to the table first, and then the other arm moves it to the bin
            category = random.choice(list(cfg.categories.keys()))
            objects = random.sample(cfg.categories[category], 2)
            object_locations = [(objects[0], 'left'), (objects[1], 'right')]
            move_object = random.choice(objects)
            arm = 'left arm' if move_object == objects[0] else 'right arm'
            other_arm = 'left arm' if arm == 'right arm' else 'right arm'
            true_action = (
                arm, move_object, 'table center', other_arm, move_object,
                'left' if move_object == objects[1] else 'right'
            )
            task = task.replace('{category}', category)
            ambiguous = True

        elif task == 'Move the {object} to the bin.':
            objects = random.sample(all_objects, 1)
            object_locations = [(objects[0], 'table center')]
            side = random.choice(['left', 'right'])
            true_action = (side + ' arm', objects[0], side)
            task = task.replace('{object}', objects[0])
            ambiguous = True

        elif task == 'Move the {object} to the {side} bin.':
            side = random.choice(['left', 'right'])
            objects = random.sample(all_objects, 1)
            object_locations = [(objects[0], 'table center')]
            true_action = (side + ' arm', objects[0], side)
            task = task.replace('{object}', objects[0]).replace('{side}', side)
            ambiguous = False

        elif task == 'Place the {category1} next to the {category2}.':
            category1 = random.choice(list(cfg.categories.keys()))
            object_category1 = random.sample(cfg.categories[category1], 2)
            category2 = random.choice(list(cfg.categories.keys()))
            object_category2 = random.choice(cfg.categories[category2])
            object_locations = [(object_category1[0], 'left'),
                                (object_category1[1], 'right'),
                                (object_category2, 'table center')]
            move_object = random.choice(object_category1)
            arm = 'left arm' if move_object == object_category1[
                0] else 'right arm'
            true_action = (arm, move_object, 'table center offset')
            task = task.replace('{category1}',
                                category1).replace('{category2}', category2)
            ambiguous = True

        elif task == 'Move the nearest object in the bins to the center.':
            objects = random.sample(all_objects, 2)
            far_close_order = ['far', 'close']
            random.shuffle(far_close_order)
            object_locations = [(objects[0], 'left ' + far_close_order[0]),
                                (objects[1], 'right ' + far_close_order[1])]
            close_object = objects[0] if far_close_order[
                0] == 'close' else objects[1]
            arm = 'left arm' if close_object == objects[0] else 'right arm'
            true_action = (arm, close_object, 'table center')
            ambiguous = False

        elif task == 'Move the farthest object in the bins to the center.':
            objects = random.sample(all_objects, 2)
            far_close_order = ['far', 'close']
            random.shuffle(far_close_order)
            object_locations = [(objects[0], 'left ' + far_close_order[0]),
                                (objects[1], 'right ' + far_close_order[1])]
            far_object = objects[0] if far_close_order[
                0] == 'far' else objects[1]
            arm = 'left arm' if far_object == objects[0] else 'right arm'
            true_action = (arm, far_object, 'table center')
            ambiguous = False

        elif task == 'Move the object at the corner of the table to the {side} bin.':
            objects = random.sample(all_objects, 3)
            side = random.choice(['left', 'right'])
            locations = random.sample([
                'table top left', 'table top right', 'table bottom left',
                'table bottom right'
            ], 3)
            object_locations = [(objects[0], locations[0]),
                                (objects[1], locations[1]),
                                (objects[2], locations[2])]
            move_object = random.choice(objects)
            arm = 'left arm' if side == 'left' else 'right arm'
            true_action = (arm, move_object, side)
            task = task.replace('{side}', side)
            ambiguous = True

        elif task == 'Move the object at the {corner} corner of the table to the {side} bin.':
            side = random.choice(['left', 'right'])
            corner = random.choice([
                'top left', 'top right', 'bottom left', 'bottom right'
            ])
            object_locations = [
                (random.choice(all_objects), 'table ' + corner)
            ]
            arm = 'left arm' if side == 'left' else 'right arm'
            true_action = (arm, object_locations[0][0], side)
            task = task.replace('{side}', side).replace('{corner}', corner)
            ambiguous = False

        elif task == 'Move the centermost object on the table to the {side} bin.':
            side = random.choice(['left', 'right'])
            corners = random.sample([
                'table top left', 'table top right', 'table bottom left',
                'table bottom right'
            ], 2)
            locations = corners + ['table center']
            objects = random.sample(all_objects, 3)
            object_locations = [(objects[0], locations[0]),
                                (objects[1], locations[1]),
                                (objects[2], locations[2])]
            arm = 'left arm' if side == 'left' else 'right arm'
            true_action = (arm, objects[2], side)
            task = task.replace('{side}', side)
            ambiguous = False

        elif task == 'Move {object} to the empty bin.':
            empty_side = random.choice(['left', 'right'])
            objects = random.sample(all_objects, 2)
            object_locations = [
                (objects[0], 'table center'),
                (objects[1], 'left' if empty_side == 'right' else 'right')
            ]
            arm = 'left arm' if empty_side == 'left' else 'right arm'
            true_action = (arm, objects[0], empty_side)
            task = task.replace('{object}', objects[0])
            ambiguous = False

        else:
            raise 'Unknown task template!'

        # Add more distractor objects
        distractor_locations = []
        if task_cfg.distractor:
            num_extra_obj = cfg.num_target_obj - len(object_locations)
            possible_distractor_locations = [
                loc for loc in cfg.locations
                if loc not in task_cfg.distractor_exclude_locations
            ]
            locations = random.sample(
                possible_distractor_locations, num_extra_obj
            )
            # find the categories of objects that are not in the task
            distractor_possible_categories = [
                category for category in cfg.categories if all([
                    obj not in cfg.categories[category]
                    for obj, loc in object_locations
                ])
            ]
            distractor_categories = random.sample(
                distractor_possible_categories, num_extra_obj
            )
            for category in distractor_categories:
                distractor_locations.append(
                    (random.choice(cfg.categories[category]), locations.pop())
                )

        # Get scene description
        # left, right, left far, left close, right far, right close, table, table top left, table top right, table bottom left, table bottom right
        scene = 'There is '
        all_locations = object_locations + distractor_locations
        random.shuffle(all_locations)
        for obj, loc in all_locations:
            if loc == 'left':
                scene += f'a {obj} in the left bin, '
            elif loc == 'right':
                scene += f'a {obj} in the right bin, '
            elif loc == 'left far':
                scene += f'a {obj} on the far side of the left bin, '
            elif loc == 'left close':
                scene += f'a {obj} on the close side of the left bin, '
            elif loc == 'right far':
                scene += f'a {obj} on the far side of the right bin, '
            elif loc == 'right close':
                scene += f'a {obj} on the close side of the right bin, '
            elif loc == 'table center':
                scene += f'a {obj} at the center of the table, '
            elif loc == 'table top left':
                scene += f'a {obj} at the top left corner of the table, '
            elif loc == 'table top right':
                scene += f'a {obj} at the top right corner of the table, '
            elif loc == 'table bottom left':
                scene += f'a {obj} at the bottom left corner of the table, '
            elif loc == 'table bottom right':
                scene += f'a {obj} at the bottom right corner of the table, '
            else:
                raise 'Unknown location!'
        scene = scene[:-2] + '.'

        # Save data
        init_data_all.append({
            'task': task,
            'object_locations': object_locations,
            'distractor_locations': distractor_locations,
            'true_action': true_action,
            'scene_description': scene,
        })

        # Log
        logging.info('============ Data {} ============'.format(data_ind + 1))
        logging.info(f'Task: {task}')
        logging.info(f'Object and location: {object_locations}')
        logging.info(f'Distractor and location: {distractor_locations}')
        logging.info(f'True action: {true_action}')
        logging.info(f'Scene description: {scene}')
        logging.info(f'Ambiguous? {ambiguous}')
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