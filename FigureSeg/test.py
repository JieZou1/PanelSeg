import tensorflow as tf
import logging
import core

flags = tf.app.flags
flags.DEFINE_string('list_path', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\eval.txt',
                    'path to list file.')
flags.DEFINE_string('tfrecord_path', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\eval_tf.record',
                    'Path to output TFRecord file.')
FLAGS = flags.FLAGS


def test_create_figure_dataset():
    figure_set = core.FigureSet()
    # figure_set.create_figure_set_from_list(list_file=FLAGS.list_path)
    figure_set.figure_set_from_tfrecord(FLAGS.tfrecord_path)


def main(_):
    test_create_figure_dataset()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.enable_eager_execution()
    print("TensorFlow version: {}".format(tf.VERSION))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    tf.app.run()
