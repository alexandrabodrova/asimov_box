"""
Tabletop manipulation simulation in PyBullet.

Run final actions in the tabletop rearrangement task, single-step.

In practice, we will have the final actions from LLM, load them here, eval the API function, and execute the action. The action is already labeled as planning success or not in previous script. Here we just record execution success or not.

# box and xyzmap are aligned
# box follows image convention: the first and third coordinates are the y coordinates (down), and the second and fourth coordinates are the x coordinates (right)
# xyzmap has first axis with down as positive y, and second axis as right as positive x. The robot is at origin.
# xyzmap boundaries: top left - [-0.3 -0.2  0. ]; top right - [0.3 -0.2  0. ]; bottom left - [-0.3 -0.8 0.00621685]; bottom right - [0.3 -0.8 0.00547853]

"""
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio
import itertools
from omegaconf import OmegaConf

from env.pick_place_env import PickPlaceEnv
from env.detection.vild import ViLD

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add a flag for using gui or not
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Whether to use PyBullet GUI or not",
    )
    parser.add_argument(
        "--load_data_path",
        type=str,
        default="data",
        help="Path to previous pickle data file",
    )
    parser.add_argument(
        "--save_data_path",
        type=str,
        default="data",
        help="Path to save data file",
    )
    args = parser.parse_args()

    # Initialize simulation environment
    camera_param = OmegaConf.create()
    camera_param.noise = False
    env = PickPlaceEnv(render=args.gui, camera_param=camera_param)
    np.random.seed(42)

    # Initialize object detector
    detector = ViLD()
    # text_embedding_path = 'env/test/text_features.npz'
    text_embedding_path = None

    # Load data
    # with open(args.data_path, 'rb') as f:
    #     data_all = pickle.load(f)
    # request, scene_description, plan_success, final_mc
    data_all = [{}]

    # Run all actions
    for data_ind, data in enumerate(data_all):
        print('============ Trial {} ============'.format(data_ind))

        # Assume the fixed set of objects
        object_all = [
            'blue block',
            'green block',
            'yellow block',
            'blue bowl',
            'green bowl',
            'yellow bowl',
        ]
        object_names = [
            'block',
            'bowl',
        ]
        color_names = [
            'blue',
            'green',
            'yellow',
        ]
        category_names = itertools.product(color_names, object_names)
        category_names = [f'{c} {o}' for c, o in category_names]
        prompt_swaps = [
            ('block', 'cube'),
            ('bowl', 'dish'),
        ]

        # Final multiple choice - e.g., 'move yellow block to add_left_offset_from_obj_pos('green bowl')', 'move yellow block and blue block to add_left_offset_from_obj_pos('green bowl')', 'move yellow block to blue bowl'
        # quite hacky but should work
        data[
            'final_mc'
        ] = "move yellow bowl and blue bowl to add_front_offset_from_obj_pos('green block')"
        final_mc = data['final_mc']
        pick_obj_phrase = final_mc.split('move')[1].strip().split('to'
                                                                 )[0].strip()
        pick_obj_all = pick_obj_phrase.split('and')
        pick_obj_all = [p.strip() for p in pick_obj_all]
        place_obj_phrase = final_mc.split('to')[1].strip()
        if 'offset' in place_obj_phrase:
            place_obj = place_obj_phrase.split("('")[1].split("')")[0]
            offset = place_obj_phrase.split('add_')[1].split('_offset')[0]
        else:
            place_obj = place_obj_phrase
            offset = None
        print('Instruction:', final_mc)
        print('Pick object(s):', pick_obj_all)
        print('Place object:', place_obj)
        print('Offset:', offset)

        # Define and reset environment.
        config = {
            'obj_names': object_all,
        }
        obs = env.reset(config)
        before = env.get_camera_image()
        prev_obs = obs['image'].copy()

        # Get images - use orthographic top-view for object detection.
        fig = plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        img = env.get_camera_image()
        plt.title('Perspective side-view')
        plt.imshow(img)
        plt.subplot(1, 3, 2)
        img = env.get_camera_image_top()
        img = np.flipud(img.transpose(1, 0, 2))
        plt.title('Orthographic top-view')
        plt.imshow(img)
        image_path = 'env/test/tmp.jpg'
        imageio.imwrite(image_path, img)
        plt.subplot(1, 3, 3)
        plt.title('Unprojected orthographic top-view')
        plt.imshow(obs['image'])
        plt.show()
        img = obs['image']
        image_path = 'env/test/tmp.jpg'
        imageio.imwrite(image_path, img)

        # markdown ViLD settings.
        category_name_string = ";".join(category_names)
        max_boxes_to_draw = 6  #@param {type:"integer"}
        nms_threshold = 0.4  #@param {type:"slider", min:0, max:0.9, step:0.05}
        min_rpn_score_thresh = 0.4  #@param {type:"slider", min:0, max:1, step:0.01}
        min_box_area = 30
        max_box_area = 3000
        vild_params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area
        found_objects, boxes, text_embedding_path = detector.infer(
            image_path,
            category_name_string,
            vild_params,
            prompt_swaps=prompt_swaps,
            display_img=True,
            text_embedding_path=text_embedding_path,
        )
        print('Found objects:', found_objects)

        prev_release_height = 0
        for pick_obj in pick_obj_all:
            # Assume we already get the pick-place object and location from the action generated by LM
            # if ind == 0:
            #     pick_obj = 'green bowl'
            #     place_pos = 'blue bowl'
            #     offset = None
            # else:
            #     pick_obj = 'yellow bowl'
            #     place_pos = 'blue bowl'
            #     offset = None
            place_pos = place_obj
            try:
                pick_box = boxes[found_objects.index(pick_obj)]
            except:
                print('Could not find pick object:', pick_obj)
                continue
            try:
                place_box = boxes[found_objects.index(place_pos)]
            except:
                print('Could not find place object:', place_pos)
                continue

            # Get pick position.
            pick_yx = [
                int((pick_box[0] + pick_box[2]) / 2),
                int((pick_box[1] + pick_box[3]) / 2)
            ]
            pick_xyz = obs['xyzmap'][pick_yx[0], pick_yx[1]]

            # Get place position.
            place_yx = [
                int((place_box[0] + place_box[2]) / 2),
                int((place_box[1] + place_box[3]) / 2)
            ]
            place_xyz = obs['xyzmap'][place_yx[0], place_yx[1]]

            # apply positive x offset if picking up bowl to grasp the edge
            if 'bowl' in pick_obj:
                pick_xyz[0] += 0.04
                place_xyz[0] += 0.04
            if 'bowl' in pick_obj and 'bowl' in place_pos:
                place_xyz[2] = prev_release_height + 0.07
            elif 'block' in pick_obj and 'bowl' in place_pos:
                place_xyz[2] = prev_release_height + 0.05
            elif 'block' in pick_obj and 'block' in place_pos:
                place_xyz[2] = prev_release_height + 0.04
            elif 'bowl' in pick_obj and 'block' in place_pos:
                place_xyz[2] = prev_release_height + 0.06

            # apply offset to left/right/front/back
            # TODO: account for possible collision with other objects
            if offset is not None:
                place_xyz[2] = prev_release_height + 0.05  # placing on table
                if offset == 'left':
                    place_xyz[0] -= 0.1
                elif offset == 'right':
                    place_xyz[0] += 0.1
                elif offset == 'front':
                    place_xyz[1] -= 0.1
                elif offset == 'back':
                    place_xyz[1] += 0.1

            # record release height
            prev_release_height = place_xyz[2]

            # Step environment.
            act = {
                'pick': pick_xyz,
                'place': place_xyz,
                'rotate_for_bowl': 'bowl' in pick_obj
            }
            obs, _, _, _ = env.step(act)

        # # Show pick and place action.
        # plt.imshow(prev_obs)
        # plt.arrow(
        #     pick_yx[1],
        #     pick_yx[0],
        #     place_yx[1] - pick_yx[1],
        #     place_yx[0] - pick_yx[0],
        #     color='w',
        #     head_starts_at_zero=False,
        #     head_width=7,
        #     length_includes_head=True,
        # )
        # plt.show()

        # Show camera image after pick and place.
        plt.subplot(1, 2, 1)
        plt.title('Before')
        plt.imshow(before)
        plt.subplot(1, 2, 2)
        plt.title('After')
        after = env.get_camera_image()
        plt.imshow(after)
        plt.show()
