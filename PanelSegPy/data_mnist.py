import argparse
import gzip
import os
from six.moves import urllib

import shutil
import tensorflow as tf
import numpy as np

DATA_DIRECTORY = 'Z:\\datasets\\MNIST'
MODEL_DIRECTORY = 'z:\\models\\MNIST'


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        num_images = read32(f)
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic, f.name))
        if rows != 28 or cols != 28:
            raise ValueError('Invalid MNIST file %s: Expected 28x28 images, found %dx%d' % (f.name, rows, cols))


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        num_items = read32(f)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic, f.name))


def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
    zipped_filepath = filepath + '.gz'
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


def dataset(directory, images_file, labels_file):
    images_file = download(directory, images_file)
    labels_file = download(directory, labels_file)

    check_image_file_header(images_file)
    check_labels_file_header(labels_file)

    def decode_image(image):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        return image / 255.0

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
        return tf.to_int32(label)

    images = tf.data.FixedLengthRecordDataset(images_file, 28 * 28, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(labels_file, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))


def train_input_fn(batch_size=32, buffer_size=50000, train_epochs=40):
    """tf.data.Dataset object for MNIST training data."""
    train_set = dataset(DATA_DIRECTORY, 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
    train_set.cache().shuffle(buffer_size=buffer_size).batch(batch_size).repeat(train_epochs)
    return train_set


def test_input_fn(batch_size=32):
    """tf.data.Dataset object for MNIST test data."""
    test_set = dataset(DATA_DIRECTORY, 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
    test_set = test_set.batch(batch_size)  # Batch size to use
    return test_set


def test_input():
    train_set = train_input_fn(32)
    iterator = train_set.make_one_shot_iterator()

    with tf.Session() as sess:
        first_batch = sess.run(iterator.get_next())
    print(first_batch)

    test_set = test_input_fn(32)
    iterator = test_set.make_one_shot_iterator()
    with tf.Session() as sess:
        first_batch = sess.run(iterator.get_next())
    print(first_batch)


def model_fn(features, labels, mode, params):
    data_format = params['data_format']

    image = features
    if isinstance(image, dict):
        image = features['image']

    if data_format == 'channels_first':
        input_shape = [-1, 1, 28, 28]
    else:
        assert data_format == 'channels_last'
        input_shape = [-1, 28, 28, 1]

    training = True
    if mode == tf.estimator.ModeKeys.PREDICT:
        training = False
    elif mode == tf.estimator.ModeKeys.EVAL:
        training = False

    y = tf.reshape(image, input_shape)
    y = tf.layers.Conv2D(32, 5, padding='same', data_format=data_format, activation=tf.nn.relu)(y)
    y = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format)(y)
    y = tf.layers.Conv2D(64, 5, padding='same', data_format=data_format, activation=tf.nn.relu)(y)
    y = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format)(y)
    y = tf.layers.flatten(y)
    y = tf.layers.Dense(1024, activation=tf.nn.relu)(y)
    y = tf.layers.Dropout(0.4)(y, training=training)
    y = tf.layers.Dense(10)(y)

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = y
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

        # If we are running multi-GPU, we need to wrap the optimizer.
        if params.get('multi_gpu'):
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        logits = y
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))
        # Name the accuracy tensor 'train_accuracy' to demonstrate the LoggingTensorHook.
        tf.identity(accuracy[1], name='train_accuracy')
        tf.summary.scalar('train_accuracy', accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    if mode == tf.estimator.ModeKeys.EVAL:
        logits = y
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1)), })


def test_classification(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int,
                        help='batch size')
    parser.add_argument('--train_steps', default=10000, type=int,
                        help='number of training steps')
    parser.add_argument('--train_epochs', type=int, default=40,
                        help='Number of epochs to train.')
    parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'],
                        help='A flag to override the data format used in the model. '
                        'channels_first provides a performance boost on GPU but is not always '
                        'compatible with CPU.')
    parser.add_argument('--export_dir', type=str, default=MODEL_DIRECTORY,
                        help='The directory where the exported SavedModel will be stored.')
    args = parser.parse_args(argv[1:])

    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=MODEL_DIRECTORY,
        params={
            'data_format': args.data_format,
            'batch_size': args.batch_size,
            'train_epochs': args.train_epochs,
        })

    tensors_to_log = {'train_accuracy': 'train_accuracy'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    mnist_classifier.train(input_fn=lambda: train_input_fn(), hooks=[logging_hook])

    eval_results = mnist_classifier.evaluate(input_fn=test_input_fn)
    print()
    print('Evaluation results:\n\t%s' % eval_results)

    # Export the model
    if args.export_dir is not None:
        image = tf.placeholder(tf.float32, [None, 28, 28])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({'image': image, })
        mnist_classifier.export_savedmodel(args.export_dir, input_fn)


def main(argv):
    # test_input()
    test_classification(argv)
    pass


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
