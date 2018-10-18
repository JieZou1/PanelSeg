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

import time

from figure.misc import CLASS_LABEL_MAPPING, case_same_label, iou, intersection_area, union
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


def _remove_overlapping_boxes(image, n_classes, image_detections, image_boxes, image_scores, image_labels):
    def overlap_with_mask(box, mask):
        roi = mask[box[1]:box[3], box[0]:box[2]]
        count = np.count_nonzero(roi)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        ratio = count / box_area
        return ratio

    def update_mask(box, mask):
        mask[box[1]:box[3], box[0]:box[2]] = 1

    idxs = list()
    height, width, depth = image.shape
    masks = list()
    for i in range(n_classes):
        masks.append(np.zeros((height, width)))

    for idx, box in enumerate(image_boxes):
        box_i = box.astype(int)
        mask = masks[image_labels[idx]]
        if overlap_with_mask(box_i, mask) < 0.66:
            idxs.append(idx)
            update_mask(box_i, mask)

    image_detections, image_boxes, image_scores, image_labels = image_detections[idxs], image_boxes[idxs], image_scores[idxs], image_labels[idxs]
    return image_detections, image_boxes, image_scores, image_labels


def _remove_overlapping_labels(label_detections, label_boxes, label_scores, label_labels):
    indexes = list()
    for i, box in enumerate(label_boxes):
        overlapped = False
        for index in indexes:
            collected_box = label_boxes[index]
            if intersection_area(box, collected_box) > 0:
                overlapped = True
                break
        if not overlapped:
            indexes.append(i)

    label_detections = label_detections[indexes]
    label_boxes = label_boxes[indexes]
    label_scores = label_scores[indexes]
    label_labels = label_labels[indexes]
    return label_detections, label_boxes, label_scores, label_labels


def _panel_label_distance(panel_box, label_box):
    """
    Computing the distance between a panel and a label
    :param panel_box:
    :param label_box:
    :return:
    """
    panel_box_l, panel_box_r, panel_box_t, panel_box_b = panel_box[0], panel_box[2], panel_box[1], panel_box[3]
    label_box_l, label_box_r, label_box_t, label_box_b = label_box[0], label_box[2], label_box[1], label_box[3]
    if label_box_r < panel_box_l:
        h_dist = panel_box_l - label_box_r
    elif label_box_l > panel_box_r:
        h_dist = label_box_l - panel_box_r
    else:
        h_dist = 0
    if label_box_b < panel_box_t:
        v_dist = panel_box_t - label_box_b
    elif label_box_t > panel_box_b:
        v_dist = label_box_t - panel_box_b
    else:
        v_dist = 0

    return h_dist + v_dist


def _is_qualified_path(label_boxes, label_scores, label_labels, label_indexes):

    # check whether there are duplicates
    labels = list()
    for index in label_indexes:
        if index == -1: # No label is assigned to the panel
            continue
        label = label_labels[index]
        if label in labels:
            return False
        else:
            labels.append(label)
    # labels stores the labels selected so far in the path

    # check whether same case (Upper, Lower, Digits)
    label_chars = list()
    numeric, upper_case, lower_case = False, False, False
    for label in labels:
        label_char = CLASS_LABEL_MAPPING[label]
        label_chars.append(label_char)
        if case_same_label(label_char):
            continue
        if label_char.isnumeric():
            numeric = True
        elif label_char.isupper():
            upper_case = True
        elif label_char.islower():
            lower_case = True
    # label_chars stores the label chars selected so far in the path

    count = 0
    if numeric is True:
        count += 1
    if upper_case is True:
        count += 1
    if lower_case is True:
        count += 1

    if count > 1:
        return False

    # label boxes can not overlap
    boxes = list()
    for index in label_indexes:
        if index == -1: # No label is assigned to the panel
            continue
        label_box = label_boxes[index]
        for box in boxes:
            if intersection_area(box, label_box) > 0:
                return False
        boxes.append(label_box)
    # boxs stores the label boxes selected so far in the path

    # check whether boxes of later labels are in the upper-left region of front labels
    # label_chars must be in the same case, otherwise, they should be rejected already
    # We also just need to check the last one, the previous ones must in the order already
    last_label_char = label_chars[-1].upper()
    last_label_box = boxes[-1]
    for i in range(len(boxes) - 1):
        label_char = label_chars[i].upper()
        label_box = boxes[i]
        if label_char < last_label_char:
            # label_box can not be in the lower-right quadrant of last_label_char
            if label_box[2] > last_label_box[0] and label_box[3] > last_label_box[1]:
                return False
        else:
            if last_label_box[2] > label_box[0] and last_label_box[3] > label_box[1]:
                return False

    # TODO: Check other qualifications

    return True


def _merge_unlabeled_panels(panel_boxes, panel_scores, panel_labels, winning_label_indexes):
    # Merge unlabelled panels to labeled neighbors. Overlapping first (larger, better) then distance (smaller, better).
    panel_indexes_assigned = list()
    panel_indexes_not_assigned = list()
    label_indexes_assigned = list()
    for i in range(len(panel_boxes)):
        if winning_label_indexes[i] == -1:    # no-label assigned yet
            panel_indexes_not_assigned.append(i)
        else:   # the panel has been assigned a label
            panel_indexes_assigned.append(i)
            label_indexes_assigned.append(winning_label_indexes[i])

    # merge panel_indexes_not_assigned to panel_indexes_assigned
    panel_indexes_to_merge = list()
    for panel_index_not_assigned in panel_indexes_not_assigned:
        # check overlapping
        panel_not_assigned_box = panel_boxes[panel_index_not_assigned]
        max_area = -1
        max_index = -1
        for panel_index_assigned in panel_indexes_assigned:
            panel_assigned_box = panel_boxes[panel_index_assigned]
            area = intersection_area(panel_assigned_box, panel_not_assigned_box)
            if area > max_area:
                max_area = area
                max_index = panel_index_assigned
        if max_area > 0:  # overlapped, we merge
            panel_indexes_to_merge.append(max_index)
            continue

        # No overlapping labeled panel found, we find the minimum distance
        min_dist = 3000  # just a big number
        min_index = -1
        for panel_index_assigned in panel_indexes_assigned:
            panel_assigned_box = panel_boxes[panel_index_assigned]
            dist = _panel_label_distance(panel_assigned_box, panel_not_assigned_box)
            if dist < min_dist:
                min_dist = dist
                min_index = panel_index_assigned
        panel_indexes_to_merge.append(min_index)

    for i, panel_index_to_merge in enumerate(panel_indexes_to_merge):
        panel0 = panel_boxes[panel_index_to_merge]
        panel1 = panel_boxes[panel_indexes_not_assigned[i]]
        panel = union(panel0, panel1)
        panel_boxes[panel_index_to_merge] = panel

    panel_boxes = panel_boxes[panel_indexes_assigned]
    panel_scores = panel_scores[panel_indexes_assigned]
    panel_labels = panel_labels[panel_indexes_assigned]
    panel_detections = np.concatenate(
        [panel_boxes, np.expand_dims(panel_scores, axis=1), np.expand_dims(panel_labels, axis=1)], axis=1)

    return panel_detections, panel_boxes, panel_scores, panel_labels


def _assign_labels_to_panels(panel_detections, panel_boxes, panel_scores, panel_labels,
                             label_detections, label_boxes, label_scores, label_labels):
    # calculate distances
    distances = list()
    for panel_box in panel_boxes:
        dists = list()
        for label_box in label_boxes:
            dist = _panel_label_distance(panel_box, label_box)
            dist = round(dist)
            dists.append(dist)
        distances.append(dists)

    # Beam search
    beam_length = 100
    all_item_pairs = []  # in the format (overall_distance, label_indexes)
    for panel_i in range(len(panel_boxes)):
        panel_width = panel_boxes[panel_i][2] - panel_boxes[panel_i][0]
        panel_height = panel_boxes[panel_i][3] - panel_boxes[panel_i][1]
        item_pairs = []
        if panel_i == 0:
            for label_i in range(len(label_boxes)):
                dist = distances[panel_i][label_i]  # dist as the primary value for sorting
                # we do not allow the distance to be larger than the 1/2 of panel side
                if dist > panel_width / 2 or dist > panel_height/ 2:
                    continue
                # score as the second value for sorting, so we divide it by 1000, to make sure that it can't be larger than 1
                score = (panel_scores[panel_i] + label_scores[label_i]) / 1000
                label_indexes = [label_i]
                item_pair = [dist - score, label_indexes]
                item_pairs.append(item_pair)
        else:
            prev_item_pairs = all_item_pairs[panel_i - 1]
            for pair_i, prev_item_pair in enumerate(prev_item_pairs):
                prev_label_indexes = prev_item_pair[1]
                prev_dist = prev_item_pair[0]
                for label_i in range(len(label_boxes)):
                    if label_i in prev_label_indexes:
                        continue  # We allow a label assigned to one panel only
                    dist = distances[panel_i][label_i] + prev_dist  # dist as the primary value for sorting
                    # we do not allow the distance to be larger than the 1/2 of panel side
                    if dist > panel_width / 2 or dist > panel_height / 2:
                        continue
                    # score as the second value for sorting, so we divide it by 1000, to make sure that it can't be larger than 1
                    score = (panel_scores[panel_i] + label_scores[label_i]) / 1000
                    label_indexes = list(prev_item_pair[1])
                    label_indexes.append(label_i)
                    item_pair = [dist - score, label_indexes]
                    if _is_qualified_path(label_boxes, label_scores, label_labels, label_indexes):
                        item_pairs.append(item_pair)

        # we add a case, when no label should be assigned to this panel
        if panel_i == 0:
            item_pairs.append([0, [-1]])
        else:
            for pair_i, prev_item_pair in enumerate(prev_item_pairs):
                label_indexes = list(prev_item_pair[1])
                label_indexes.append(-1)  # We use -1 to indicate there is no label assigned to this panel
                dist = prev_item_pair[0]
                item_pair = [dist, label_indexes]
                item_pairs.append(item_pair)

        # sort item_pairs for both distance and combined scores
        item_pairs.sort(key=lambda pair: pair[0])

        # keep only at most beam_length item pairs
        if len(item_pairs) > beam_length:
            item_pairs = item_pairs[:beam_length]
        all_item_pairs.append(item_pairs)

    # ToDO: remove all [-1] path?? May not be necessary, since the winners will be sorted according to scores too.

    # check the last item_pairs
    winners = list()
    min_dist = all_item_pairs[-1][0][0]
    for dist, label_indexes in all_item_pairs[-1]:   # we check only the last column
        if dist - min_dist < 1: # collect the smallest distance only
            winners.append(label_indexes)
        else:
            break
    # same distances, sort according to their overall scores, and then the winner is with the highest score.
    score_index_pairs = []  # in the format (overall_score, winner_indexes)
    for winner in winners:
        score = 0
        for panel_index, label_index in enumerate(winner):
            if label_index == -1:
                continue
            score += (label_scores[label_index] + panel_scores[panel_index])
        score_index_pairs.append([-score, winner])
    score_index_pairs.sort(key=lambda pair: pair[0])

    winning_indexes = score_index_pairs[0][1]   # stores the indexes of the labels to be assigned to the panel

    # Remove unlabeled panels if they significantly overlap labeled panels
    indexes = list()
    for i, panel_box in enumerate(panel_boxes):
        if winning_indexes[i] == -1:    # no-label assigned yet
            area = (panel_box[2] - panel_box[0])*(panel_box[3] - panel_box[1])
            overlapped_area = 0
            for j, box in enumerate(panel_boxes):
                if winning_indexes[j] != -1:
                    overlapped_area += intersection_area(box, panel_box)
            percent = overlapped_area / area
            if percent < 0.5:
                indexes.append(i)
        else:
            indexes.append(i)

    panel_detections = panel_detections[indexes]
    panel_boxes = panel_boxes[indexes]
    panel_scores = panel_scores[indexes]
    panel_labels = panel_labels[indexes]
    winning_label_indexes = [winning_indexes[i] for i in indexes]  # remove the corresponding winning_indexes too

    # tried merging unlabeled panels (not helpful much)
    # panel_detections, panel_boxes, panel_scores, panel_labels = _merge_unlabeled_panels(panel_boxes, panel_scores, panel_labels, winning_label_indexes)

    # assign labels to panels, such that we could evaluate as the complete panel segmentation
    # for i in range(len(panel_boxes)):
    #     if winning_label_indexes[i] != -1:
    #         label = CLASS_LABEL_MAPPING[label_labels[winning_label_indexes[i]]]

    # Remove labels which are not in the winning labels indexes
    indexes = [index for index in winning_label_indexes if index != -1]
    label_detections = label_detections[indexes]
    label_boxes = label_boxes[indexes]
    label_scores = label_scores[indexes]
    label_labels = label_labels[indexes]
    # re-assign winning indexes
    k = 0
    for i in range(len(winning_label_indexes)):
        if winning_label_indexes[i] == -1:
            winning_label_indexes[i] = -1
        else:
            winning_label_indexes[i] = k
            k += 1

    return panel_detections, panel_boxes, panel_scores, panel_labels, \
           label_detections, label_boxes, label_scores, label_labels, winning_label_indexes


def _merge_panel_label_detections(panel_detections, panel_boxes, panel_scores, panel_labels,
                                  label_detections, label_boxes, label_scores, label_labels):
    """
    Use beam search to assign labels to panels according to the overall distances.
    When panal and label overlap, the distance is 0. Otherwise is the manhattan distance
    If the overall distance is the same, we pick the path, which has the highest overall score
    We do not allow duplicated labels in the path, we do not allow mixed cases (upper, lower, or digits)
    One panel can have one label only
    :param image_detections:
    :param image_boxes:
    :param image_scores:
    :param image_labels:
    :param image_l_detections:
    :param image_l_boxes:
    :param image_l_scores:
    :param image_l_labels:
    :return:
    """

    # assign labels to panels and remove unassigned labels
    if len(label_detections) > 0:
        # assign labels to panels
        panel_detections, panel_boxes, panel_scores, panel_labels, \
            label_detections, label_boxes, label_scores, label_labels, \
            panel_label_indexes = _assign_labels_to_panels(panel_detections, panel_boxes, panel_scores, panel_labels,
                                                           label_detections, label_boxes, label_scores, label_labels)
    else:
        panel_label_indexes = [-1]*len(panel_boxes)

    # remove unsigned low score panels (Do NOT HELP MUCH)
    # indexes = [i for i in range(len(panel_detections)) if panel_label_indexes[i] != -1 or panel_scores[i] > 0.6]
    # panel_detections = panel_detections[indexes]
    # panel_boxes = panel_boxes[indexes]
    # panel_scores = panel_scores[indexes]
    # panel_labels = panel_labels[indexes]
    # panel_label_indexes = [panel_label_indexes[i] for i in indexes]

    return panel_detections, panel_boxes, panel_scores, panel_labels, \
           label_detections, label_boxes, label_scores, label_labels, panel_label_indexes


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
    all_panel_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_label_detections = [[None for i in range(generator.l_num_classes())] for j in range(generator.size())]
    all_merge_detections = [[None for i in range(generator.l_num_classes() + 1)] for j in range(generator.size())]

    mean_time = 0

    for i in range(generator.size()):
        # if i != 18:
        #     continue
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network to predict
        start = time.time()

        boxes, scores, labels, l_boxes, l_scores, l_labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:6]

        processing_time = time.time() - start
        mean_time += processing_time

        # remove low score detections and sort all detections according to their scores.
        panel_detections, panel_boxes, panel_scores, panel_labels = _get_all_detections(boxes, scores, labels, scale, score_threshold, max_detections)
        label_detections, label_boxes, label_scores, label_labels = _get_all_detections(l_boxes, l_scores, l_labels, scale, l_score_threshold, l_max_detections)

        # remove boxes which overlaps with other high score boxes (NOT SEEM USEFUL)
        # panel_detections, panel_boxes, panel_scores, panel_labels = _remove_overlapping_boxes(raw_image, generator.num_classes(), panel_detections, panel_boxes, panel_scores, panel_labels)

        # merge panel and label detection
        if len(panel_detections) > 0:  # panel detected
            panel_detections, panel_boxes, panel_scores, panel_labels, \
            label_detections, label_boxes, label_scores, label_labels, panel_label_indexes = \
                _merge_panel_label_detections(panel_detections, panel_boxes, panel_scores, panel_labels,
                                              label_detections, label_boxes, label_scores, label_labels)
            #  create merged results
            merge_boxes, merge_scores, merge_labels = panel_boxes.copy(), panel_scores.copy(), panel_labels.copy()
            for k in range(len(merge_labels)):
                if panel_label_indexes[k] == -1:
                    merge_labels[k] = generator.l_num_classes()
                else:
                    merge_labels[k] = label_labels[panel_label_indexes[k]]

        else:   # no panel detected
            label_detections, label_boxes, label_scores, label_labels = _remove_overlapping_labels(
                label_detections, label_boxes, label_scores, label_labels)
            merge_boxes, merge_scores, merge_labels = panel_boxes.copy(), panel_scores.copy(), panel_labels.copy()

        merge_detections = np.concatenate(
            [merge_boxes, np.expand_dims(merge_scores, axis=1), np.expand_dims(merge_labels, axis=1)], axis=1)

        if save_path is not None:
            # draw_annotations(raw_image, generator.load_annotations(i),
            #                  label_to_name=generator.label_to_name, l_label_to_name=generator.l_label_to_name)
            draw_detections(raw_image, panel_boxes, panel_scores, panel_labels,
                            label_to_name=generator.label_to_name, score_threshold=score_threshold)
            draw_detections(raw_image, label_boxes, label_scores, label_labels,
                            label_to_name=generator.l_label_to_name, score_threshold=l_score_threshold)

            cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_panel_detections[i][label] = panel_detections[panel_detections[:, -1] == label, :-1]
        for label in range(generator.l_num_classes()):
            all_label_detections[i][label] = label_detections[label_detections[:, -1] == label, :-1]
        for label in range(generator.l_num_classes() + 1):
            all_merge_detections[i][label] = merge_detections[merge_detections[:, -1] == label, :-1]

        print('{}/{}, processing time {}'.format(i + 1, generator.size(), processing_time), end='\r')

    print('average processing time: {0}'.format(mean_time / generator.size()))

    return all_panel_detections, all_label_detections, all_merge_detections


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
    all_panel_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_label_annotations = [[None for i in range(generator.l_num_classes())] for j in range(generator.size())]
    all_merge_annotations = [[None for i in range(generator.l_num_classes() + 1)] for j in range(generator.size())]

    for i in range(generator.size()):
        # load the annotations
        annotations = generator.load_annotations(i)
        panel_annotations = annotations[:, :5]
        label_annotations = annotations[:, 5:]
        merge_annotations = annotations[:, [0, 1, 2, 3, -1]].copy()
        for k in range(len(merge_annotations)):
            if merge_annotations[k][-1] == -1:
                merge_annotations[k][-1] = generator.l_num_classes()

        # copy detections to all_panel_annotations
        for label in range(generator.num_classes()):
            all_panel_annotations[i][label] = panel_annotations[panel_annotations[:, 4] == label, :4].copy()
        for label in range(generator.l_num_classes()):
            all_label_annotations[i][label] = label_annotations[label_annotations[:, 4] == label, :4].copy()
        for label in range(generator.l_num_classes() + 1):
            all_merge_annotations[i][label] = merge_annotations[merge_annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_panel_annotations, all_label_annotations, all_merge_annotations


def evaluate(
    generator,
    model,
    panel_iou_threshold=0.5,
    panel_score_threshold=0.05,
    panel_max_detections=100,
    label_iou_threshold=0.5,
    label_score_threshold=0.05,
    label_max_detections=100,
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
    all_panel_annotations, all_label_annotations, all_merge_annotations = _get_annotations(generator)
    all_panel_detections, all_label_detections, all_merge_detections = _get_detections(generator, model,
                                                                                       score_threshold=panel_score_threshold, max_detections=panel_max_detections,
                                                                                       l_score_threshold=label_score_threshold, l_max_detections=label_max_detections,
                                                                                       save_path=save_path)

    # all_panel_detections = pickle.load(open('all_panel_detections.pkl', 'rb'))
    # all_panel_annotations = pickle.load(open('all_panel_annotations.pkl', 'rb'))
    # pickle.dump(all_panel_detections, open('all_panel_detections.pkl', 'wb'))
    # pickle.dump(all_panel_annotations, open('all_panel_annotations.pkl', 'wb'))

    # mAP evaluation
    panel_aps = eval_average_precisions(generator, generator.num_classes(), all_panel_detections, all_panel_annotations, panel_iou_threshold)
    label_aps = eval_average_precisions(generator, generator.l_num_classes(), all_label_detections, all_label_annotations, label_iou_threshold)
    merge_aps = eval_average_precisions(generator, generator.l_num_classes() + 1, all_merge_detections, all_merge_annotations, panel_iou_threshold)

    # precision recall evaluation
    panel_prs = eval_precision_recall(generator, generator.num_classes(), all_panel_detections, all_panel_annotations, panel_iou_threshold)
    label_prs = eval_precision_recall(generator, generator.l_num_classes(), all_label_detections, all_label_annotations, label_iou_threshold)
    merge_prs = eval_precision_recall(generator, generator.l_num_classes() + 1, all_merge_detections, all_merge_annotations, panel_iou_threshold)

    return panel_aps, panel_prs, label_aps, label_prs, merge_aps, merge_prs


def eval_precision_recall(generator, num_classes, all_detections, all_annotations, iou_threshold):
    precisions_recalls = {}
    # process detections and annotations
    for label in range(num_classes):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0
        num_detections = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            num_detections      += detections.shape[0]
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

        correct = true_positives.sum()
        precisions_recalls[label] = correct, num_detections, num_annotations;

        # # no annotations -> AP for this class is 0 (is this correct?)
        # if num_annotations == 0:
        #     precisions_recalls[label] = 0, 0, 0
        #     continue
        #
        # if true_positives.shape[0] == 0:
        #     precisions_recalls[label] = 0, 0, num_annotations
        #     continue
        #
        # precisions_recalls[label] = correct / true_positives.shape[0], correct / num_annotations, num_annotations

    return precisions_recalls


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
