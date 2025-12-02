""" Multi-step-multi-label, human collaboration, tabletop manipulation environment

Generate initial state in the multi-step setting in the tabletop manipulation environment. No partial observability (i.e., stacks).

Actions:
    - We want to define ground truth action(s) for each step. There could be multi-label, e.g., "Sort the blocks by color at L1 and L2", and it is okay to put the first block at either L1 or L2.
    - However, it is difficult to define the action sequence here because the true action in the following steps may depend on the action in the previous step. Thus, we define it step-by-step when collecting mc_post data after LM generates multiple choices.

Tasks:
1. Sort the things I like and I dislike in the plates/bowls.
    - I don't like eggplants or milk.
2. Preprare the ingredients for maing a breakfast/lunch.
3. Prepare the ingredients for making a 
    - Hawaiian/cheese/chicken pizza.
    - cake.
    - salad.
    - burger.
    - sandwich.
"""
import os
import argparse
import random
import pickle
import logging
from omegaconf import OmegaConf


def main(cfg):
    # object_all = list(cfg.objects.keys())
    # categories = cfg.categories
    # locations_all = cfg.locations
    # location_ratio = np.array(cfg.location_ratio) / sum(cfg.location_ratio)

    # Possible task requests
    # task_request_template = 'I {preference}. Can you help me sort the items in the {location_1} and {location_2}?'
    task_request_template = "Can you put things I like in the {loc_1}, and things I don't like in the {loc_2}?"

    # Sample all init data
    init_data_all = []
    for _ in range(cfg.num_data):

        # Sample location
        loc_like, loc_dislike = ['blue plate', 'green plate']
        # loc_like, loc_dislike = random.sample(locations_all, k=2)
        request = task_request_template.replace('{loc_1}', loc_like).replace(
            '{loc_2}', loc_dislike
        )

        # Set categories for like and dislike
        # category_name_for_preference = [
        #     'veggie', 'fruit', 'snack', 'baked good', 'meat'
        # ]
        # category_name_like, category_name_dislike = random.sample(
        #     category_name_for_preference, k=2
        # )
        # category_like = categories[category_name_like]
        # category_dislike = categories[category_name_dislike]
        category_like = [obj for obj in cfg.likes]
        category_dislike = [obj for obj in cfg.dislikes]
        # preference_indicator_context = random.choice(
        #     ['like', 'dislike']
        # )  # either list things that like or dislike in the context
        # if preference_indicator_context == 'like':
        #     category_context = category_like
        # else:
        #     category_context = category_dislike

        # Sample objects for target - at least 1 object obviously like/dislike, and last object can be obvious or ambiguous - also avoid perception conflict
        while 1:
            object_like = random.choice(
                category_like
            )  # possible the object already mentioned in the context
            object_dislike = random.choice(category_dislike)
            # flag_distractor_ambiguous = random.random(
            # ) < cfg.ratio_distractor_ambiguous
            while 1:  # make sure the distractor is not the same as the others
                # if flag_distractor_ambiguous:
                object_distract = random.choice(
                    category_like + category_dislike
                )
                # else:
                #     object_distract = random.choice(object_all)
                if object_distract not in [object_like, object_dislike]:
                    break
            objects_target = [object_like, object_dislike, object_distract]
            objects_target_like = [
                obj for obj in objects_target if obj in category_like
            ]
            objects_target_dislike = [
                obj for obj in objects_target if obj in category_dislike
            ]

            # Shuffle object order
            random.shuffle(objects_target)

            # Avoid perception conflict
            flag_conflict = False
            for conflict in cfg.perception_conflict:
                if len(set(conflict).intersection(objects_target)) > 1:
                    flag_conflict = True
                    break
            if not flag_conflict:
                break

        # Sample objects in the context - either 1 or 2 (with 1, not much information, can be more ambiguous)
        # num_object_context = random.choice(cfg.num_object_context_possible)
        # objects_context = random.sample(category_context, k=num_object_context)
        objects_context_like = random.sample(
            category_like, k=cfg.num_object_context_possible
        )
        objects_context_dislike = random.sample(
            category_dislike, k=cfg.num_object_context_possible
        )
        # if preference_indicator_context == 'like':
        #     # like + context
        #     preference = 'like' + ' ' + ' and '.join(objects_context)
        # else:
        #     preference = 'don\'t like' + ' ' + ' and '.join(objects_context)
        # request = request.replace('{preference}', preference)
        thought = 'I know that you like ' + ' and '.join(
            objects_context_like
        ) + ', and dislike ' + ' and '.join(objects_context_dislike) + '.'
        request = request.replace('{thought}', thought)

        # Get scene description
        scene_description = 'On the table there is ' + ', '.join(
            objects_target[:-1]
        ) + ', and ' + objects_target[-1] + '.'

        # Log
        logging.info(
            '=============== Data {} ==============='.
            format(len(init_data_all) + 1)
        )
        logging.info('Task: {}'.format(request))
        logging.info('Scene description: {}'.format(scene_description))
        logging.info('Thought: {}'.format(thought))
        logging.info(
            'Objects to be moved (like): {}'.format(objects_target_like)
        )
        logging.info(
            'Objects to be moved (dislike): {}'.format(objects_target_dislike)
        )
        logging.info('================= END =================\n')

        # Save data
        data = {}
        data['init'] = {
            'request': request,
            'scene_description': scene_description,
            'thought': thought,
            'objects': objects_target,
            'objects_target_like': objects_target_like,
            'objects_target_dislike': objects_target_dislike,
            'location_like': loc_like,
            'location_dislike': loc_dislike,
        }
        data['action'] = [
        ]  # initialize action executed - e.g., action.append(['we', 'blue block', 'left']) for human placing blue block on the left side (on the table, or in the bowl/plate)
        init_data_all.append(data)

    # Save all data
    with open(cfg.data_save_path, 'wb') as f:
        pickle.dump(init_data_all, f)

    # Print summary
    print("============ Summary ================")
    print('Number of data: {}'.format(len(init_data_all)))
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