import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe

flags = tf.app.flags
flags.DEFINE_string('TRAIN_LIST', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\train.txt',
                    help='the file list for training figures')
flags.DEFINE_string('EVAL_LIST', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\eval.txt',
                    help='the file list for evaluation figures')
FLAGS = flags.FLAGS


def dataset(figure_file_list):

    pass


def train_dataset():
    """
    tf.data.Dataset object for Panel training data.
    original figure images are listed in TRAIN_LIST
    dataset has also been shuffled
    """
    dataset(FLAGS.TRAIN_LIST)




def main(_):
    train_ds = train_dataset()
    features, labels = tfe.Iterator(train_ds).next()
    print("example features:", features[0])
    print("example labels:", labels[0])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.enable_eager_execution()
    print("TensorFlow version: {}".format(tf.VERSION))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    tf.app.run()
