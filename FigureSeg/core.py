"""Core Classes and Functions for FigureSeg."""
import tensorflow as tf
import tensorflow.contrib.eager as tfe
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
                    label_text = map_label(label_text)
                    label_texts.append(label_text.encode('utf8'))
                    label_classes.append(LABEL_CLASS_MAPPING[label_text])
                    label_xmins.append(panel.label_rect[0]/self.image_width)
                    label_ymins.append(panel.label_rect[1]/self.image_height)
                    label_xmaxs.append(panel.label_rect[2]/self.image_width)
                    label_ymaxs.append(panel.label_rect[3]/self.image_height)
                else:
                    label_text = map_label(label_text)
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


class FigureSet:
    """
    A class for a FigureSet
    image_path is the path to the figure image file
    id is the unique id to each figure
    image_orig is the original color image
    panels contain all panels
    """

    def __init__(self):
        self.list_file = None
        self.files = None

    def load_list(self, list_file):
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

    def convert_to_tfrecord(self, tfrecord_path):
        writer = tf.python_io.TFRecordWriter(tfrecord_path)
        for idx, file in enumerate(self.files):
            # if idx != 107:
            #     continue
            logging.info('Convert to TFRecord for image %d: %s.', idx, file)

            try:
                if not os.path.exists(file):
                    raise FigureSegError('Could not find %s.'.format(file))
                figure = Figure()
                figure.load_image(file)

                xml_path = os.path.join(figure.image_path.replace('.jpg', '_data.xml'))
                if not os.path.exists(xml_path):
                    raise FigureSegError('Could not find %s.'.format(xml_path))
                figure.load_annotation_iphotodraw(xml_path)

                tf_example = figure.to_tf_example()
            except FigureSegError as ex:
                logging.warning(ex.message)
                continue

            writer.write(tf_example.SerializeToString())

        writer.close()

    def figure_set_from_tfrecord(self, tfrecord_path):

        ds = tf.data.TFRecordDataset(tfrecord_path)
        for x in ds:
            print(x)

        slim = tf.contrib.slim

        keys_to_features = {
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/image': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/panel/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/panel/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/panel/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/panel/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/label/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/label/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/label/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/label/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/panel/label/text': tf.VarLenFeature(dtype=tf.string),
            'image/panel/label/class': tf.VarLenFeature(dtype=tf.int64),
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.Tensor('image/image'),
            'panel/bbox': slim.tfexample_decoder.BoundingBox(
                ['xmin', 'ymin', 'xmax', 'ymax'], 'image/panel/bbox/'),
            'label/bbox': slim.tfexample_decoder.BoundingBox(
                ['xmin', 'ymin', 'xmax', 'ymax'], 'image/label/bbox/'),
            'panel/text': slim.tfexample_decoder.Tensor('image/panel/label/text'),
            'panel/class': slim.tfexample_decoder.Tensor('image/panel/label/class'),
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

        dataset = slim.dataset.Dataset(
            data_sources=tfrecord_path,
            reader=tf.TFRecordReader,
            num_samples=3,  # 手动生成了三个文件， 每个文件里只包含一个example
            decoder=decoder,
            items_to_descriptions={})

        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=3,
            shuffle=False)

        [image, panel_bboxes, label_bboxes, panel_text, panel_class] = \
            provider.get(['image', 'panel/bbox', 'label/bbox', 'panel/text', 'panel/class'])

        # figure_set = tf.data.TFRecordDataset(tfrecord_path)
        # figure_set.map(self.parse_tfrecord)
        # example = tfe.Iterator(figure_set).next()

    ########################
    # Codes below are trial codes, not ready yet
    def create_figure(self, image_path):
        if not os.path.exists(image_path):
            raise FigureSegError('Could not find %s.'.format(image_path))
        figure = Figure()
        figure.load_image(image_path)

        xml_path = os.path.join(figure.image_path.replace('.jpg', '_data.xml'))
        if not os.path.exists(xml_path):
            raise FigureSegError('Could not find %s.'.format(xml_path))
        figure.load_annotation_iphotodraw(xml_path)

    def create_figure_set_from_list(self, list_file):
        self.list_file = list_file
        ds = tf.data.TextLineDataset(list_file)
        for x in ds:
            print(x)

        figure_set = tf.data.TextLineDataset(list_file).map(self.create_figure)
        figure = tfe.Iterator(figure_set).next()


    def parse_tfrecord(self, example):
        feature_dict = {
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/filename': tf.VarLenFeature(tf.string),
            'image/image': tf.FixedLenFeature([], tf.string),
            'image/panel/bbox/xmin': tf.FixedLenFeature([], tf.float32),
            'image/panel/bbox/ymin': tf.FixedLenFeature([], tf.float32),
            'image/panel/bbox/xmax': tf.FixedLenFeature([], tf.float32),
            'image/panel/bbox/ymax': tf.FixedLenFeature([], tf.float32),
            'image/label/bbox/xmin': tf.FixedLenFeature([], tf.float32),
            'image/label/bbox/ymin': tf.FixedLenFeature([], tf.float32),
            'image/label/bbox/xmax': tf.FixedLenFeature([], tf.float32),
            'image/label/bbox/ymax': tf.FixedLenFeature([], tf.float32),
            'image/panel/label/text': tf.FixedLenFeature([], tf.string),
            'image/panel/label/class': tf.FixedLenFeature([], tf.int64)
        }
        features = tf.parse_single_example(example, features=feature_dict)


    def read_from_tfrecord(tfrecord_path):
        with tf.Session() as sess:
            # create feature to hold data
            feature_dict = {
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/filename': tf.VarLenFeature(tf.string),
                'image/image': tf.FixedLenFeature([], tf.string),
                'image/panel/bbox/xmin': tf.FixedLenFeature([], tf.float32),
                'image/panel/bbox/ymin': tf.FixedLenFeature([], tf.float32),
                'image/panel/bbox/xmax': tf.FixedLenFeature([], tf.float32),
                'image/panel/bbox/ymax': tf.FixedLenFeature([], tf.float32),
                'image/label/bbox/xmin': tf.FixedLenFeature([], tf.float32),
                'image/label/bbox/ymin': tf.FixedLenFeature([], tf.float32),
                'image/label/bbox/xmax': tf.FixedLenFeature([], tf.float32),
                'image/label/bbox/ymax': tf.FixedLenFeature([], tf.float32),
                'image/panel/label/text': tf.FixedLenFeature([], tf.string),
                'image/panel/label/class': tf.FixedLenFeature([], tf.int64)
            }
            # create a list of tfrecord filenames and pass it to a queue
            filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=1)

            # define a reader and read the next record
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            # decode the record read by the reader
            features = tf.parse_single_example(serialized_example, features=feature_dict)

            image_height = tf.case(features['image/height'], tf.int64)
            image_width = tf.case(features['image/width'], tf.int64)

            # convert the image data from string back to the numbers
            image = tf.decode_raw(features['image/image'], tf.uint8)
            image = tf.reshape(image, [image_height, image_width, 3])




class FigureSegError(Exception):
    """
    Exception for FigureSeg
    """
    def __init__(self, message):
        self.message = message
