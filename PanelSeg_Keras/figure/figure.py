import os
import cv2
import logging
import xml.etree.ElementTree as ET
from figure import misc
from figure.panel import Panel
import tensorflow as tf


class Figure:
    """
    A class for a Figure
    image_path is the path to the figure image file
    id is the unique id to each figure
    image_orig is the original color image
    panels contain all panels
    """
    def __init__(self, image_path):
        self.image_path = image_path
        self.panels = None
        self.image = None
        self.image_width = 0
        self.image_height = 0

    def load_image(self):
        img = cv2.imread(self.image_path)  # BGR image, we need to convert it to RGB image
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image_height, self.image_width = self.image.shape[:2]

    def load_annotation_iphotodraw(self, annotation_file_path):
        """
        Load iPhotoDraw annotation
        """
        # create element tree object
        tree = ET.parse(annotation_file_path)
        # get root element
        root = tree.getroot()

        shape_items = root.findall('./Layers/Layer/Shapes/Shape')

        # Read All Items (Panels and Labels)
        panel_items = []
        label_items = []
        for shape_item in shape_items:
            text_item = shape_item.find('./BlockText/Text')
            text = text_item.text.lower()
            if text.startswith('panel'):
                panel_items.append(shape_item)
            elif text.startswith('label'):
                label_items.append(shape_item)
            else:
                logging.error('{0}: has unknown <shape> xml items {1}'.format(annotation_file_path, text))

        # Extract information from and validate all panel items
        # panels type: Panel
        panels = []
        for panel_item in panel_items:
            text_item = panel_item.find('./BlockText/Text')
            label_text = text_item.text
            label_text = label_text.strip()
            words = label_text.split(' ')
            if len(words) > 2:
                logging.error('{0}: {1} is not correct'.format(annotation_file_path, label_text))
                continue
            elif len(words) == 2:
                label_text = words[1]
                if len(label_text) is not 1:  # Now we process single character panel label only
                    logging.warning('{0}: panel {1} is not single character'.format(annotation_file_path, label_text))
            else:
                label_text = ''

            x_min, y_min, x_max, y_max = misc.extract_bbox_from_iphotodraw_node(panel_item, self.image_width, self.image_height)
            if x_max <= x_min or y_max <= y_min:
                logging.error('{0}: panel {1} rect is not correct!'.format(annotation_file_path, label_text))
                continue
            panel_rect = [x_min, y_min, x_max, y_max]
            panel = Panel(label_text, panel_rect, None)
            panels.append(panel)

        # Extract information from and validate all label items
        # labels type: Panel
        labels = []
        for label_item in label_items:
            text_item = label_item.find('./BlockText/Text')
            label_text = text_item.text
            label_text = label_text.strip()
            words = label_text.split(' ')
            if len(words) is not 2:
                logging.error('{0}: {1} is not correct'.format(annotation_file_path, label_text))
                continue
            label_text = words[1]
            if len(label_text) is not 1:  # Now we process single character panel label only
                logging.warning('{0}: label {1} is not single character'.format(annotation_file_path, label_text))

            x_min, y_min, x_max, y_max = misc.extract_bbox_from_iphotodraw_node(label_item, self.image_width, self.image_height)
            if x_max <= x_min or y_max <= y_min:
                logging.error('{0}: label {1} rect is not correct!'.format(annotation_file_path, label_text))
                continue
            label_rect = [x_min, y_min, x_max, y_max]
            label = Panel(label_text, None, label_rect)
            labels.append(label)

        if len(labels) != 0 and len(labels) != len(panels):
            logging.warning('{0}: has different panel and label rects. Most likely there are mixes with-label and without-label panels'.format(annotation_file_path))

        # collect all panel label characters
        char_set = set()
        for panel in panels:
            if len(panel.label) == 0:
                continue
            char_set.add(panel.label)

        # build panel dictionary according to labels
        panel_dict = {s: [] for s in char_set}
        for panel in panels:
            if len(panel.label) == 0:
                continue
            panel_dict[panel.label].append(panel)

        # build label dictionary according to labels
        label_dict = {s: [] for s in char_set}
        for label in labels:
            label_dict[label.label].append(label)

        # assign labels to panels
        for label_char in char_set:
            if len(panel_dict[label_char]) != len(label_dict[label_char]):
                logging.error('{0}: panel {1} dont have same matching labels!'.format(annotation_file_path, label_char))
                continue
            misc.assign_labels_to_panels(panel_dict[label_char], label_dict[label_char])

        # expand the panel_rect to always include label_rect
        for panel in panels:
            if panel.label_rect is not None:
                panel.panel_rect = misc.union(panel.label_rect, panel.panel_rect)

        self.panels = panels

    def save_preview(self, folder):
        """
        Save the annotation preview at folder.
        """
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        path, file = os.path.split(self.image_path)
        path = os.path.join(folder, file)
        img = self.image.copy()
        for i, panel in enumerate(self.panels):
            color = colors[i%len(colors)]
            cv2.rectangle(img, (panel.panel_rect[0], panel.panel_rect[1]), (panel.panel_rect[2], panel.panel_rect[3]), color, 3)
            if panel.label_rect is not None:
                cv2.rectangle(img, (panel.label_rect[0], panel.label_rect[1]), (panel.label_rect[2], panel.label_rect[3]), color, 2)
                cv2.putText(img, panel.label, (panel.label_rect[2] + 10, panel.label_rect[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
        cv2.imwrite(path, img)

    def to_tf_example(self):
        """
        Convert to TfRecord Example
        """
        panel_xmins = []
        panel_ymins = []
        panel_xmaxs = []
        panel_ymaxs = []
        label_xmins = []
        label_ymins = []
        label_xmaxs = []
        label_ymaxs = []
        label_texts = []
        label_classes = []
        for panel in self.panels:
            panel_xmins.append(panel.panel_rect[0]/self.image_width)
            panel_ymins.append(panel.panel_rect[1]/self.image_height)
            panel_xmaxs.append(panel.panel_rect[2]/self.image_width)
            panel_ymaxs.append(panel.panel_rect[3]/self.image_height)

            if panel.label_rect is not None:
                label_text = panel.label
                if len(label_text) == 1:    # We handle 1 char label only for now.
                    label_text = misc.map_label(label_text)
                    label_texts.append(label_text.encode('utf8'))
                    label_classes.append(misc.LABEL_CLASS_MAPPING[label_text])
                    label_xmins.append(panel.label_rect[0]/self.image_width)
                    label_ymins.append(panel.label_rect[1]/self.image_height)
                    label_xmaxs.append(panel.label_rect[2]/self.image_width)
                    label_ymaxs.append(panel.label_rect[3]/self.image_height)
                else:
                    label_text = misc.map_label(label_text)
                    label_texts.append(label_text.encode('utf8'))
                    label_classes.append(-1)    # We handle 1 char label only for now.
                    label_xmins.append(panel.label_rect[0]/self.image_width)
                    label_ymins.append(panel.label_rect[1]/self.image_height)
                    label_xmaxs.append(panel.label_rect[2]/self.image_width)
                    label_ymaxs.append(panel.label_rect[3]/self.image_height)
            else:
                label_texts.append('NONE'.encode('utf8'))
                label_classes.append(-1)
                label_xmins.append(0)
                label_xmaxs.append(0)
                label_ymaxs.append(0)
                label_ymins.append(0)

        feature_dict = {
            'image/height': misc.int64_feature(self.image_height),
            'image/width': misc.int64_feature(self.image_width),
            'image/filename': misc.bytes_feature(self.image_path.encode('utf8')),
            'image/image': misc.bytes_feature(tf.compat.as_bytes(self.image.tostring())),
            'image/panel/bbox/xmin': misc.float_list_feature(panel_xmins),
            'image/panel/bbox/ymin': misc.float_list_feature(panel_ymins),
            'image/panel/bbox/xmax': misc.float_list_feature(panel_xmaxs),
            'image/panel/bbox/ymax': misc.float_list_feature(panel_ymaxs),
            'image/label/bbox/xmin': misc.float_list_feature(label_xmins),
            'image/label/bbox/ymin': misc.float_list_feature(label_ymins),
            'image/label/bbox/xmax': misc.float_list_feature(label_xmaxs),
            'image/label/bbox/ymax': misc.float_list_feature(label_ymaxs),
            'image/panel/label/text': misc.bytes_list_feature(label_texts),
            'image/panel/label/class': misc.int64_list_feature(label_classes),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example

    def from_tf_example(self, features):
        pass
