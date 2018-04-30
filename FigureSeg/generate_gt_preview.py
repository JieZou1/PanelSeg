import logging
import tensorflow as tf
import os
import core


flags = tf.app.flags
flags.DEFINE_string('list_path', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\all.txt',
                    'the list file of the figure dataset.')
FLAGS = flags.FLAGS


def main(_):
    list_path = FLAGS.list_path
    figure_set = core.FigureSet(list_path)

    figure_set.save_gt_preview()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
