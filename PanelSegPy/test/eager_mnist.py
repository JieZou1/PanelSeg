
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from dataset import dataset_mnist


flags = tf.app.flags
flags.DEFINE_float('lr', 0.001, help='learning rate')
flags.DEFINE_float('momentum', 0.5, help='SGD momentum')
FLAGS = flags.FLAGS


keras_layers = tf.keras.layers
keras_models = tf.keras.models


class ModelCNN2(keras_models.Model):
    def __init__(self, data_format):
        super(ModelCNN2, self).__init__()

        if data_format == 'channels_first':
            input_shape = [1, 28, 28]
        else:
            assert data_format == 'channels_last'
            input_shape = [28, 28, 1]

        self.reshape = keras_layers.Reshape(input_shape)
        self.conv2d_1 = keras_layers.Conv2D(32, (5, 5), padding='same', data_format=data_format, activation='relu')
        self.maxpooling_1 = keras_layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format)
        self.conv2d_2 = keras_layers.Conv2D(64, (5, 5), padding='same', data_format=data_format, activation='relu')
        self.maxpooling_2 = keras_layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format)
        self.flatten = keras_layers.Flatten()
        self.dense1 = keras_layers.Dense(1024, activation='relu')
        self.dropout = keras_layers.Dropout(0.4)
        self.dense2 = keras_layers.Dense(10, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        result = self.reshape(inputs)
        result = self.conv2d_1(result)
        result = self.maxpooling_1(result)
        result = self.conv2d_2(result)
        result = self.maxpooling_2(result)
        result = self.flatten(result)
        result = self.dense1(result)
        result = self.dropout(result)
        result = self.dense2(result)
        return result


class ModelDense3(keras_models.Model):
    def __init__(self):
        super(ModelDense3, self).__init__()
        self.dense1 = keras_layers.Dense(10)
        self.dense2 = keras_layers.Dense(10)

    def call(self, inputs, training=None, mask=None):
        result = self.dense1(inputs)
        result = self.dense2(result)
        result = self.dense2(result)    # reuse variables from dense2 layer
        return result


def loss(model, x, y):
    y_ = model(x)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


def train():
    train_ds = dataset_mnist.train_dataset(batch_size=64, buffer_size=50000, train_epochs=40)

    # model = ModelDense3()     # 3 dense layer classifier

    # 'channels_first' is typically faster on GPUs while 'channels_last' is typically faster on CPUs.
    # See https://www.tensorflow.org/performance/performance_guide#data_formats
    model = ModelCNN2('channels_first')  # 2 layer CNN classifier

    # generate a random sample to test the model
    x = tf.zeros([1, 1, 784])
    y = model(x)
    print(y)
    model.summary()

    optimizer = tf.train.MomentumOptimizer(FLAGS.lr, FLAGS.momentum)

    loss_avg = tfe.metrics.Mean()
    accu_avg = tfe.metrics.Accuracy()

    for (i, (x, y)) in enumerate(tfe.Iterator(train_ds)):
        grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

        if (i + 1) % 100 == 0:
            loss_avg(loss(model, x, y))
            accu_avg(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
            print("Loss at step {:04d}: {:.3f}".format(i + 1, loss_avg.result()))
            print("Accuracy at step {:04d}: {:.3f}".format(i + 1, accu_avg.result()))


def main(_):
    train()

    test_ds = dataset_mnist.test_dataset()


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.logging.set_verbosity(tf.logging.INFO)
    print("TensorFlow version: {}".format(tf.VERSION))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    tf.app.run()
