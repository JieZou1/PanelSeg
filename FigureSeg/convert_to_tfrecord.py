import tensorflow as tf
import logging
import core

flags = tf.app.flags
flags.DEFINE_string('list_path', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\eval.txt',
                    'path to list file.')
flags.DEFINE_string('tfrecord_path', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\eval_tf.record',
                    'Path to output TFRecord file.')
FLAGS = flags.FLAGS


def write_to_record():
    list_path = FLAGS.list_path
    output_path = FLAGS.tfrecord_path

    figure_set = core.FigureSet()
    figure_set.load_list(list_path)
    figure_set.convert_to_tfrecord(output_path)


def read_from_record():
    tfrecord_path = FLAGS.tfrecord_path
    # core.FigureSet.read_from_tfrecord(tfrecord_path)
    # core.FigureSet.figure_set_from_tfrecord(tfrecord_path)
    figure_set = core.FigureSet()
    figure_set.figure_set_from_tfrecord(tfrecord_path)


def main(_):
    write_to_record()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.enable_eager_execution()
    print("TensorFlow version: {}".format(tf.VERSION))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    tf.app.run()
