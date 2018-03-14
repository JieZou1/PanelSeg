import argparse
import os
import tensorflow as tf

DATA_DIRECTORY = 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\'
MODEL_DIRECTORY = 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\models\\label_classification_2\\'


def read_image(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    image_resized = tf.cast(image_resized, tf.float32)
    image_resized = image_resized / 255.0
    return image_resized, label


def dataset(label_folder, non_label_folder):

    # Read filenames
    all_label_files, all_non_label_files = [], []
    folders = [dI for dI in os.listdir(label_folder) if os.path.isdir(os.path.join(label_folder, dI))]
    for f in folders:
        folder = os.path.join(label_folder, f)
        src_file = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.png')]
        all_label_files += src_file

    all_non_label_files = [os.path.join(non_label_folder, file) for file in os.listdir(non_label_folder) if file.endswith('.png')]

    filenames = all_label_files + all_non_label_files
    labels = [[0]]*len(all_label_files) + [[1]]*len(all_non_label_files)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(read_image)
    return dataset


def train_input_fn(batch_size=32, buffer_size=50000, train_epochs=40):
    train_set = dataset(DATA_DIRECTORY + 'Labels-28', DATA_DIRECTORY + 'NonLabels-28')
    train_set.cache().shuffle(buffer_size=buffer_size).batch(batch_size).repeat(train_epochs)
    return train_set


def model_fn(features, labels, mode, params):

    image = features
    training = True
    if mode == tf.estimator.ModeKeys.PREDICT:
        training = False
    elif mode == tf.estimator.ModeKeys.EVAL:
        training = False

    input_shape = [-1, 28, 28, 3]

    y = tf.reshape(image, input_shape)
    y = tf.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu, name='conv1')(y)
    y = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same', name='pool1')(y)
    y = tf.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu, name='conv2')(y)
    y = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same', name='pool2')(y)
    y = tf.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu, name='conv3')(y)
    y = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same', name='pool3')(y)

    y = tf.layers.flatten(y, name='flat1')
    y = tf.layers.Dense(128, activation=tf.nn.relu, name='dense1')(y)
    y = tf.layers.Dropout(0.4)(y, training=training)
    y = tf.layers.Dense(2, name='output')(y)

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
        optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)

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


def test_input():
    train_set = train_input_fn(32)
    iterator = train_set.make_one_shot_iterator()

    with tf.Session() as sess:
        first_batch = sess.run(iterator.get_next())
    print(first_batch)


def test_train(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int,
                        help='batch size')
    parser.add_argument('--train_steps', default=10000, type=int,
                        help='number of training steps')
    parser.add_argument('--train_epochs', type=int, default=40,
                        help='Number of epochs to train.')
    parser.add_argument('--export_dir', type=str, default=MODEL_DIRECTORY,
                        help='The directory where the exported SavedModel will be stored.')
    args = parser.parse_args(argv[1:])

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=MODEL_DIRECTORY,
        params={
            'batch_size': args.batch_size,
            'train_epochs': args.train_epochs,
        })

    tensors_to_log = {'train_accuracy': 'train_accuracy'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    classifier.train(input_fn=lambda: train_input_fn(), hooks=[logging_hook])

    # Export the model !!! Some problems with saving the model !!!
    if args.export_dir is not None:
        image = tf.placeholder(tf.float32, [None, 28, 28])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({'image': image, })
        classifier.export_savedmodel(args.export_dir, input_fn)


def main(argv):
    # test_input()
    test_train(argv)
    pass


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
