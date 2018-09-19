"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations

import keras
import numpy as np
import os

import cv2


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _merge_panel_label_detections(image_detections, image_boxes, image_scores, image_labels,
                                  image_l_detections, image_l_boxes, image_l_scores, image_l_labels):

    # remove duplicates of labels
    # indexes = list()
    # values = set()
    # for index, l in enumerate(image_l_labels):
    #     len0 = len(values)
    #     values.add(l)
    #     len1 = len(values)
    #     if len1 > len0:
    #         indexes.append(index)
    # image_l_detections = image_l_detections[indexes]
    # image_l_boxes = image_l_boxes[indexes]
    # image_l_scores = image_l_scores[indexes]
    # image_l_labels = image_l_labels[indexes]
    return image_detections, \
           image_boxes, \
           image_scores, \
           image_labels, \
           image_l_detections, \
           image_l_boxes, \
           image_l_scores, \
           image_l_labels


def _get_detections(generator, model,
                    score_threshold=0.05, max_detections=100,
                    l_score_threshold=0.05, l_max_detections=100,
                    save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_l_detections = [[None for i in range(generator.l_num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network to predict
        boxes, scores, labels, l_boxes, l_scores, l_labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:6]

        # remove low score detections and sort all detections acoording th their scores.
        image_detections, image_boxes, image_scores, image_labels = _get_all_detections(
            boxes, scores, labels, scale, score_threshold, max_detections)
        image_l_detections, image_l_boxes, image_l_scores, image_l_labels = _get_all_detections(
            l_boxes, l_scores, l_labels, scale, l_score_threshold, l_max_detections)

        # merge panel and label detection
        image_detections, image_boxes, image_scores, image_labels, \
        image_l_detections, image_l_boxes, image_l_scores, image_l_labels = _merge_panel_label_detections(
            image_detections, image_boxes, image_scores, image_labels,
            image_l_detections, image_l_boxes, image_l_scores, image_l_labels)

        if save_path is not None:
            # draw_annotations(raw_image, generator.load_annotations(i),
            #                  label_to_name=generator.label_to_name, l_label_to_name=generator.l_label_to_name)
            draw_detections(raw_image, image_boxes, image_scores, image_labels,
                            label_to_name=generator.label_to_name, score_threshold=0.5)
            draw_detections(raw_image, image_l_boxes, image_l_scores, image_l_labels,
                            label_to_name=generator.l_label_to_name, score_threshold=0.5)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]
        for label in range(generator.l_num_classes()):
            all_l_detections[i][label] = image_l_detections[image_l_detections[:, -1] == label, :-1]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections, all_l_detections


def _get_all_detections(boxes, scores, labels, scale, score_threshold, max_detections):
    # correct boxes for image scale
    boxes /= scale

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those scores
    scores = scores[0][indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)[:max_detections]

    # select detections
    image_boxes = boxes[0, indices[scores_sort], :]
    image_scores = scores[scores_sort]
    image_labels = labels[0, indices[scores_sort]]
    image_detections = np.concatenate(
        [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

    return image_detections, image_boxes, image_scores, image_labels


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_l_annotations = [[None for i in range(generator.l_num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
        for label in range(generator.l_num_classes()):
            all_l_annotations[i][label] = annotations[annotations[:, 9] == label, 5:9].copy()

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_annotations, all_l_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    l_iou_threshold=0.5,
    l_score_threshold=0.05,
    l_max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, all_l_detections     = _get_detections(generator, model,
                                                           score_threshold=score_threshold, max_detections=max_detections,
                                                           l_score_threshold=l_score_threshold, l_max_detections=l_max_detections,
                                                           save_path=save_path)
    all_annotations, all_l_annotations    = _get_annotations(generator)

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # mAP evaluation
    average_precisions = eval_average_precisions(generator, generator.num_classes(), all_detections, all_annotations, iou_threshold)
    average_l_precisions = eval_average_precisions(generator, generator.l_num_classes(), all_l_detections, all_l_annotations, l_iou_threshold)

    # clef accuracy evaluation
    clef_accuracies_precisions_recalls = eval_clef_accuracy(generator, generator.num_classes(), all_detections, all_annotations, 0.66)
    l_clef_accuracies_precisions_recalls = eval_clef_accuracy(generator, generator.l_num_classes(), all_l_detections, all_l_annotations, 0.66)

    return average_precisions, clef_accuracies_precisions_recalls, average_l_precisions, l_clef_accuracies_precisions_recalls


def eval_clef_accuracy(generator, num_classes, all_detections, all_annotations, iou_threshold):
    clef_accuracies_precisions_recalls = {}
    # process detections and annotations
    for label in range(num_classes):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            clef_accuracies_precisions_recalls[label] = 0, 0, 0, 0
            continue

        correct = true_positives.sum()
        clef_accuracies_precisions_recalls[label] = correct / max(true_positives.shape[0], num_annotations), \
                                                    correct / true_positives.shape[0], \
                                                    correct / num_annotations, \
                                                    num_annotations

    return clef_accuracies_precisions_recalls


def eval_average_precisions(generator, num_classes, all_detections, all_annotations, iou_threshold):
    average_precisions = {}
    # process detections and annotations
    for label in range(num_classes):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions
