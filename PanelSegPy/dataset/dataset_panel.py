import tensorflow as tf


def read_sample_list(list_path):
    with tf.gfile.GFile(list_path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]
