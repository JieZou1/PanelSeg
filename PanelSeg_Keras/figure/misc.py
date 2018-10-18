import math
import numpy as np
import tensorflow as tf


LABEL_CLASS_MAPPING = {
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
    '6': 5,
    '7': 6,
    '8': 7,
    '9': 8,
    'A': 9,
    'B': 10,
    'C': 11,
    'D': 12,
    'E': 13,
    'F': 14,
    'G': 15,
    'H': 16,
    'I': 17,
    'J': 18,
    'K': 19,
    'L': 20,
    'M': 21,
    'N': 22,
    'O': 23,
    'P': 24,
    'Q': 25,
    'R': 26,
    'S': 27,
    'T': 28,
    'U': 29,
    'V': 30,
    'W': 31,
    'X': 32,
    'Y': 33,
    'Z': 34,
    'a': 35,
    'b': 36,
    'd': 37,
    'e': 38,
    'f': 39,
    'g': 40,
    'h': 41,
    'i': 42,
    'j': 43,
    'l': 44,
    'm': 45,
    'n': 46,
    'q': 47,
    'r': 48,
    't': 49,
}
CLASS_LABEL_MAPPING = {v: k for k, v in LABEL_CLASS_MAPPING.items()}


def map_label(c):
    if c == 'c' or c == 'C':
        return 'C'
    elif c == 'k' or c == 'K':
        return 'K'
    elif c == 'o' or c == 'O':
        return 'O'
    elif c == 'p' or c == 'P':
        return 'P'
    elif c == 's' or c == 'S':
        return 'S'
    elif c == 'u' or c == 'U':
        return 'U'
    elif c == 'v' or c == 'V':
        return 'V'
    elif c == 'w' or c == 'W':
        return 'W'
    elif c == 'x' or c == 'X':
        return 'X'
    elif c == 'y' or c == 'Y':
        return 'Y'
    elif c == 'z' or c == 'Z':
        return 'Z'
    else:
        return c


def case_same_label(c):
    if c == 'c' or c == 'C':
        return True
    elif c == 'k' or c == 'K':
        return True
    elif c == 'o' or c == 'O':
        return True
    elif c == 'p' or c == 'P':
        return True
    elif c == 's' or c == 'S':
        return True
    elif c == 'u' or c == 'U':
        return True
    elif c == 'v' or c == 'V':
        return True
    elif c == 'w' or c == 'W':
        return True
    elif c == 'x' or c == 'X':
        return True
    elif c == 'y' or c == 'Y':
        return True
    elif c == 'z' or c == 'Z':
        return True
    else:
        return False


def extract_bbox_from_iphotodraw_node(item, image_width, image_height):
    """
    Extract bounding box information from Element item (ElementTree).
    It also makes sure that the bounding box is within the image
    :param item:
    :param image_width:
    :param image_height:
    :return: x_min, y_min, x_max, y_max
    """
    extent_item = item.find('./Data/Extent')
    height = extent_item.get('Height')
    width = extent_item.get('Width')
    x = extent_item.get('X')
    y = extent_item.get('Y')
    x_min = round(float(x))
    y_min = round(float(y))
    x_max = x_min + round(float(width))
    y_max = y_min + round(float(height))
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > image_width:
        x_max = image_width
    if y_max > image_height:
        y_max = image_height

    return x_min, y_min, x_max, y_max


def assign_labels_to_panels(panels, labels):
    """
    Use beam search to assign labels to panels according to the overall distance
    Assign labels.label_rect to panels.label_rect
    panels and labels must have the same length
    :param panels: panels having the same label character
    :param labels: labels having the same label character
    """
    # calculate distance from panels to labels
    distances = []
    for panel_i, panel in enumerate(panels):
        dists = []
        panel_rect = panel.panel_rect
        panel_c = [(panel_rect[0] + panel_rect[2])/2.0, (panel_rect[1] + panel_rect[3])/2.0]
        for label_i, label in enumerate(labels):
            label_rect = label.label_rect
            label_c = [(label_rect[0] + label_rect[2])/2.0, (label_rect[1] + label_rect[3])/2.0]

            d = math.hypot(panel_c[0] - label_c[0], panel_c[1] - label_c[1])
            dists.append(d)
        distances.append(dists)

    # Beam search
    beam_length = 100
    all_item_pairs = []  # in the format (overall_distance, label_indexes)
    for panel_i, panel in enumerate(panels):
        item_pairs = []
        if panel_i == 0:
            for label_i, label in enumerate(labels):
                dist = distances[panel_i][label_i]
                label_indexes = [label_i]
                item_pair = [dist, label_indexes]
                item_pairs.append(item_pair)
        else:
            prev_item_pairs = all_item_pairs[panel_i - 1]
            for pair_i, prev_item_pair in enumerate(prev_item_pairs):
                prev_label_indexes = prev_item_pair[1]
                prev_dist = prev_item_pair[0]
                for label_i, label in enumerate(labels):
                    if label_i in prev_label_indexes:
                        continue  # We allow a label assigned to one panel only
                    dist = distances[panel_i][label_i] + prev_dist
                    label_indexes = list(prev_item_pair[1])
                    label_indexes.append(label_i)
                    item_pair = [dist, label_indexes]
                    item_pairs.append(item_pair)

        # sort item_pairs
        item_pairs.sort(key=lambda pair: pair[0])
        # keep only at most beam_length item pairs
        if len(item_pairs) > 100:
            item_pairs = item_pairs[:beam_length]

        all_item_pairs.append(item_pairs)

    # check the last item_pairs
    best_path = all_item_pairs[-1][0][1]
    for i in range(len(panels)):
        panels[i].label_rect = labels[best_path[i]].label_rect


def read_sample_list(list_path):
    with tf.gfile.GFile(list_path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]


def union(a, b):
    # a and b should be [x1,y1,x2,y2]
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[2], b[2]) - x
    h = max(a[3], b[3]) - y
    return [x, y, x + w, y + h]


def intersection(a, b):
    # a and b should be [x1,y1,x2,y2]
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[2], b[2]) - x
    h = min(a[3], b[3]) - y
    if w < 0 or h < 0:
        return [0, 0, 0, 0]
    return [x, y, x+w, y+h]


def union_area(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection_area(ai, bi):
    # a and b should be (x1,y1,x2,y2)
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection_area(a, b)
    area_u = union_area(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

