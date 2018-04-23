import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe

flags = tf.app.flags
flags.DEFINE_string('TRAIN_LIST', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\train.txt',
                    help='the file list for training figures')
flags.DEFINE_string('EVAL_LIST', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\eval.txt',
                    help='the file list for evaluation figures')
FLAGS = flags.FLAGS


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


def dataset(figure_file_list):

    pass


def train_dataset():
    """
    tf.data.Dataset object for Panel training data.
    original figure images are listed in TRAIN_LIST
    dataset has also been shuffled
    """
    dataset(FLAGS.TRAIN_LIST)


def test(_):
    train_ds = train_dataset()
    features, labels = tfe.Iterator(train_ds).next()
    print("example features:", features[0])
    print("example labels:", labels[0])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.enable_eager_execution()
    print("TensorFlow version: {}".format(tf.VERSION))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    tf.app.run(main='test')
