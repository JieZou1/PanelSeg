"""Core Classes and Functions for FigureSeg."""
import tensorflow as tf
import logging
import os
import xml.etree.ElementTree as ET
import cv2
import misc

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


class Panel:
    """
    A class for a Panel
    """
    def __init__(self, label, panel_rect, label_rect):
        self.label = label
        self.panel_rect = panel_rect    # list [x_min, y_min, x_max, y_max]
        self.label_rect = label_rect    # list [x_min, y_min, x_max, y_max]


class Figure:
    """
    A class for a Figure
    image_path is the path to the figure image file
    id is the unique id to each figure
    image_orig is the original color image
    panels contain all panels
    """
    def __init__(self):
        self.image_path = None
        self.panels = None
        self.image = None
        self.image_width = 0
        self.image_height = 0

    def load_image(self, image_path):
        self.image_path = image_path
        img = cv2.imread(image_path)  # BGR image, we need to convert it to RGB image
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
        xml_path = os.path.join(self.image_path.replace('.jpg', '_data.xml'))

        if not os.path.exists(xml_path):
            raise FigureSegError('Could not find %s, ignoring example.'.format(xml_path))

        self.load_annotation_iphotodraw(xml_path)

        # with tf.gfile.GFile(xml_path, 'r') as fid:
        #     xml_str = fid.read()
        # while not (xml_str.startswith('<Document') or xml_str.startswith('<document')):  # remove the first lines
        #     xml_str = xml_str.split('\n', 1)[1]
        # xml = etree.fromstring(xml_str)
        #
        # if xml_str.startswith('<document'):
        #     data = recursive_parse_xml_to_dict(xml)['document']
        # else:
        #     data = recursive_parse_xml_to_dict(xml)['Document']

        try:
            tf_example = dict_to_tf_example(data, example)
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', xml_path)

        return tf_example


class FigureSet:
    """
    A class for a FigureSet
    image_path is the path to the figure image file
    id is the unique id to each figure
    image_orig is the original color image
    panels contain all panels
    """

    def __init__(self, list_file):
        self.list_file = list_file
        self.files = misc.read_sample_list(list_file)

    def validate_annotation(self):
        for idx, file in enumerate(self.files):
            # if idx != 5:
            #     continue
            logging.info('Validate Annotation of Image %d: %s.', idx, file)

            try:
                if not os.path.exists(file):
                    raise FigureSegError('Could not find %s.'.format(file))
                figure = Figure()
                figure.load_image(file)

                xml_path = os.path.join(figure.image_path.replace('.jpg', '_data.xml'))
                if not os.path.exists(xml_path):
                    raise FigureSegError('Could not find %s.'.format(xml_path))
                figure.load_annotation_iphotodraw(xml_path)

            except FigureSegError as ex:
                logging.warning(ex.message)
                continue

    def save_gt_preview(self):
        for idx, file in enumerate(self.files):
            # if idx != 5:
            #     continue
            logging.info('Generate GT Annotation Preview for Image %d: %s.', idx, file)

            try:
                if not os.path.exists(file):
                    raise FigureSegError('Could not find %s.'.format(file))
                figure = Figure()
                figure.load_image(file)

                xml_path = os.path.join(figure.image_path.replace('.jpg', '_data.xml'))
                if not os.path.exists(xml_path):
                    raise FigureSegError('Could not find %s.'.format(xml_path))
                figure.load_annotation_iphotodraw(xml_path)

            except FigureSegError as ex:
                logging.warning(ex.message)
                continue

            # Save preview
            folder, file = os.path.split(figure.image_path)
            folder = os.path.join(folder, "prev")
            figure.save_preview(folder)


    def save_in_tfrecord(self, train_output_path):
        writer = tf.python_io.TFRecordWriter(train_output_path)
        for idx, file in enumerate(self.files):
            # if idx != 3876:
            #     continue
            logging.info('On image %d: %s.', idx, file)

            try:
                figure = Figure(file)
                tf_example = figure.to_tf_example()
            except FigureSegError as ex:
                logging.warning(ex.message)
                continue

            writer.write(tf_example.SerializeToString())

        writer.close()


class FigureSegError(Exception):
    """
    Exception for FigureSeg
    """
    def __init__(self, message):
        self.message = message
