r"""Convert the the PanelSeg dataset to TFRecord.

"""

import hashlib
import os
import logging
import random
import io

from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

logging.basicConfig(level=logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython',
                    'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython',
                    'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS


def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `Shape` tags are the only ones that can appear
    multiple times at the same level of a tree.

    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
      Python dictionary holding XML contents.
    """
    if not xml:
        return {xml.tag: xml}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'Shape':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def dict_to_tf_example(data, label_map_dict, img_path):

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = image.width
    height = image.height

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []

    for shape in data['Layers']['Layer']['Shapes']['Shape']:

        text = shape['BlockText']['Text'].text
        if not (text.startswith('Panel') or text.startswith('panel')):
            continue

        attrib = shape['Data']['Extent'].attrib
        x = float(attrib['X'])
        y = float(attrib['Y'])
        w = float(attrib['Width'])
        h = float(attrib['Height'])

        xmin = x
        xmax = x + w
        ymin = y
        ymax = y + h

        xmins.append(xmin / width)
        ymins.append(ymin / height)
        xmaxs.append(xmax / width)
        ymaxs.append(ymax / height)

        class_name = 'Panel'
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])

    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(img_path.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(img_path.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename, label_map_dict, examples):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        # if idx != 3876:
        #     continue
        logging.info('On image %d: %s.', idx, example)

        xml_path = os.path.join(example.replace('.jpg', '_data.xml'))

        if not os.path.exists(xml_path):
            logging.warning('Could not find %s, ignoring example.', xml_path)
            continue

        with tf.gfile.GFile(xml_path, 'r') as fid:
            xml_str = fid.read()
        while not (xml_str.startswith('<Document') or xml_str.startswith('<document')):  # remove the first lines
            xml_str = xml_str.split('\n', 1)[1]
        xml = etree.fromstring(xml_str)

        if xml_str.startswith('<document'):
            data = recursive_parse_xml_to_dict(xml)['document']
        else:
            data = recursive_parse_xml_to_dict(xml)['Document']

        try:
            tf_example = dict_to_tf_example(data, label_map_dict, example)
            writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', xml_path)

    writer.close()


def main(_):
    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading from dataset.')
    # image_dir = os.path.join(data_dir, 'images')
    # annotations_dir = os.path.join(data_dir, 'annotations')
    examples_path = os.path.join(data_dir, 'train.txt')
    examples_list = dataset_util.read_examples_list(examples_path)

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    logging.info('%d training and %d validation examples.', len(train_examples), len(val_examples))

    train_output_path = os.path.join(FLAGS.output_dir, 'tf_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'tf_val.record')

    create_tf_record(train_output_path, label_map_dict, train_examples)
    create_tf_record(val_output_path, label_map_dict, val_examples)


if __name__ == '__main__':
  tf.app.run()
