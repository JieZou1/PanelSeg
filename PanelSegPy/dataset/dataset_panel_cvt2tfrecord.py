import tensorflow as tf
import io
import os
import logging
from lxml import etree
import hashlib
import PIL.Image
from dataset import dataset_panel, dataset_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython',
                    'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython',
                    'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\panel_label_map.pbtxt',
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


def dict_to_tf_example(data, img_path):

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

        xmin /= width
        ymin /= height
        xmax /= width
        ymax /= height

        if xmin < 0 or ymin < 0 or xmax > 1.01 or ymax > 1.01:
            print(img_path)

        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)

        class_name = 'Panel'
        classes_text.append(class_name.encode('utf8'))
        classes.append(1)  # Panel bounding box is assigned label 1, others label 0

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


def create_tf_records(train_list_path, train_output_path):

    samples = dataset_panel.read_sample_list(train_list_path)
    writer = tf.python_io.TFRecordWriter(train_output_path)

    for idx, example in enumerate(samples):
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
            tf_example = dict_to_tf_example(data, example)
            writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', xml_path)

    writer.close()


def main(_):
    train_list_path = os.path.join(FLAGS.data_dir, 'train.txt')
    train_output_path = os.path.join(FLAGS.output_dir, 'tf_train_all.record')
    create_tf_records(train_list_path, train_output_path)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # tf.enable_eager_execution()
    # print("TensorFlow version: {}".format(tf.VERSION))
    # print("Eager execution: {}".format(tf.executing_eagerly()))
    tf.app.run()
