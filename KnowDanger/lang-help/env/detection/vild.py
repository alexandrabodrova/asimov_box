"""
Run zero-shot open-vocabulary object detection with [ViLD](https://arxiv.org/abs/2104.13921) to generate a list of objects as a scene description for a large language model.

Not using GPT for ViLD - it is somehow slower for inference with GPU and tensorflow.

"""
import os
import numpy as np
import tensorflow.compat.v1 as tf
import torch
from PIL import Image
from easydict import EasyDict
import matplotlib.pyplot as plt
from pathlib import Path
import clip
import time

from env.detection.util import nms, build_text_embedding
from env.detection.visualization import paste_instance_masks, display_image, visualize_boxes_and_labels_on_image_array


home_path = str(Path.home())

#Define ViLD hyperparameters.
FLAGS = {
    'prompt_engineering': True,
    'this_is': True,
    'temperature': 100.0,
    'use_softmax': False,
}
FLAGS = EasyDict(FLAGS)

# Parameters for drawing figure.
display_input_size = (10, 10)
overall_fig_size = (18, 24)


class ViLD():

    def __init__(self):

        # Load ViLD model
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.session = tf.Session(
            graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options)
        )
        saved_model_dir = os.path.join(
            home_path, "colab-content/image_path_v2"
        )
        _ = tf.saved_model.loader.load(
            self.session, ["serve"], saved_model_dir
        )
        numbered_categories = [{
            "name": str(idx),
            "id": idx,
        } for idx in range(50)]
        self.numbered_category_indices = {
            cat["id"]: cat for cat in numbered_categories
        }

        #Load CLIP model.
        torch.cuda.set_per_process_memory_fraction(0.9, None)
        clip_model, clip_preprocess = clip.load("ViT-B/32")
        clip_model.cuda().eval()
        # print('Using gpu')
        # print(
        #     "Model parameters:",
        #     f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}"
        # )
        # print("Input resolution:", clip_model.visual.input_resolution)
        # print("Context length:", clip_model.context_length)
        # print("Vocab size:", clip_model.vocab_size)
        self.clip_model = clip_model

    def infer(
        self,
        image_path,
        category_name_string,
        params,
        prompt_swaps=[],
        display_img=True,
        text_embedding_path=None,
    ):
        """
        Forward pass of ViLD model.
        
        """
        #################################################################
        # Preprocessing categories and get params
        for a, b in prompt_swaps:
            category_name_string = category_name_string.replace(a, b)
        category_names = [x.strip() for x in category_name_string.split(";")]
        category_names = ["background"] + category_names
        categories = [{
            "name": item,
            "id": idx + 1,
        } for idx, item in enumerate(category_names)]
        category_indices = {cat["id"]: cat for cat in categories}

        max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area = params
        fig_size_h = min(max(5, int(len(category_names) / 2.5)), 10)

        #################################################################
        # Obtain results and read image
        s1 = time.time()
        roi_boxes, roi_scores, detection_boxes, scores_unused, box_outputs, detection_masks, visual_features, image_info = self.session.run(
            [
                "RoiBoxes:0", "RoiScores:0", "2ndStageBoxes:0",
                "2ndStageScoresUnused:0", "BoxOutputs:0", "MaskOutputs:0",
                "VisualFeatOutputs:0", "ImageInfo:0"
            ], feed_dict={"Placeholder:0": [image_path,]}
        )
        # print('Inference time:', time.time() - s1)
        roi_boxes = np.squeeze(roi_boxes, axis=0)  # squeeze
        # no need to clip the boxes, already done
        roi_scores = np.squeeze(roi_scores, axis=0)

        detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
        scores_unused = np.squeeze(scores_unused, axis=0)
        box_outputs = np.squeeze(box_outputs, axis=0)
        detection_masks = np.squeeze(detection_masks, axis=0)
        visual_features = np.squeeze(visual_features, axis=0)

        image_info = np.squeeze(image_info, axis=0)  # obtain image info
        image_scale = np.tile(image_info[2:3, :], (1, 2))
        image_height = int(image_info[0, 0])
        image_width = int(image_info[0, 1])

        rescaled_detection_boxes = detection_boxes / image_scale  # rescale

        # Read image
        image = np.asarray(Image.open(open(image_path, "rb")).convert("RGB"))
        assert image_height == image.shape[0]
        assert image_width == image.shape[1]

        #################################################################
        # Filter boxes
        # print('1: ', len(detection_boxes))

        # Apply non-maximum suppression to detected boxes with nms threshold.
        nmsed_indices = nms(detection_boxes, roi_scores, thresh=nms_threshold)
        # print('2: ', len(nmsed_indices))

        # Compute RPN box size.
        box_sizes = (
            rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]
        ) * (rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])

        # Filter out invalid rois (nmsed rois)
        valid_indices = np.where(
            np.logical_and(
                np.isin(np.arange(len(roi_scores), dtype=int), nmsed_indices),
                np.logical_and(
                    np.logical_not(np.all(roi_boxes == 0., axis=-1)),
                    np.logical_and(
                        roi_scores >= min_rpn_score_thresh,
                        np.logical_and(
                            box_sizes > min_box_area, box_sizes < max_box_area
                        )
                    )
                )
            )
        )[0]
        # print('3: ', len(valid_indices))

        detection_roi_scores = roi_scores[valid_indices][:max_boxes_to_draw,
                                                         ...]
        detection_boxes = detection_boxes[valid_indices][:max_boxes_to_draw,
                                                         ...]
        detection_masks = detection_masks[valid_indices][:max_boxes_to_draw,
                                                         ...]
        detection_visual_feat = visual_features[valid_indices
                                               ][:max_boxes_to_draw, ...]
        rescaled_detection_boxes = rescaled_detection_boxes[
            valid_indices][:max_boxes_to_draw, ...]

        #################################################################
        # Compute text embeddings and detection scores, and rank results
        # if text_embedding_path is None:
        text_features = build_text_embedding(categories, self.clip_model)
        #     np.savez('env/test/text_features.npz', text_features=text_features)
        # else:
        #     text_features = np.load(text_embedding_path)['text_features']

        raw_scores = detection_visual_feat.dot(text_features.T)
        if FLAGS.use_softmax:
            scores_all = softmax(FLAGS.temperature * raw_scores, axis=-1)
        else:
            scores_all = raw_scores

        indices = np.arange(len(scores_all))
        indices_fg = np.array([
            i for i in indices if np.argmax(scores_all[i]) != 0
        ])

        #################################################################
        # Print found_objects
        found_objects = []
        for a, b in prompt_swaps:
            category_names = [
                name.replace(b, a) for name in category_names
            ]  # Extra prompt engineering.
        for anno_idx in indices[0:int(rescaled_detection_boxes.shape[0])]:
            scores = scores_all[anno_idx]
            if np.argmax(scores) == 0:  # background
                # print('No object found with score:', np.max(scores))
                continue
            found_object = category_names[np.argmax(scores)]
            # print("Found a", found_object, "with score:", np.max(scores))
            found_objects.append(found_object)

            # # print sorted scores with category names in one line
            # print(
            #     ' '.join([
            #         '{}: {:.3f}'.format(category_names[i], scores[i])
            #         for i in np.argsort(-scores)
            #     ])
            # )

        #################################################################
        # Plot detected boxes on the input image.
        if display_img:
            ymin, xmin, ymax, xmax = np.split(
                rescaled_detection_boxes, 4, axis=-1
            )
            processed_boxes = np.concatenate([
                xmin, ymin, xmax - xmin, ymax - ymin
            ], axis=-1)
            segmentations = paste_instance_masks(
                detection_masks, processed_boxes, image_height, image_width
            )

            if len(indices_fg) == 0:
                display_image(np.array(image), size=overall_fig_size)
                print(
                    "ViLD does not detect anything belong to the given category"
                )
            else:
                image_with_detections = visualize_boxes_and_labels_on_image_array(
                    np.array(image),
                    rescaled_detection_boxes[indices_fg],
                    valid_indices[:max_boxes_to_draw][indices_fg],
                    detection_roi_scores[indices_fg],
                    self.numbered_category_indices,
                    instance_masks=segmentations[indices_fg],
                    use_normalized_coordinates=False,
                    max_boxes_to_draw=max_boxes_to_draw,
                    min_score_thresh=min_rpn_score_thresh,
                    skip_scores=False,
                    skip_labels=True,
                )
                # plt.figure(figsize=overall_fig_size)
                plt.imshow(image_with_detections)
                # plt.axis("off")
                plt.title("ViLD detected objects and RPN scores.")
                plt.show()

        return found_objects, rescaled_detection_boxes, text_embedding_path
