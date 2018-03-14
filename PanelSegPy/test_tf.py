import numpy as np
import tensorflow as tf


def test1():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    linear_model = tf.layers.Dense(units=1)
    y = linear_model(x)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))


def test_regression():
    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

    linear_model = tf.layers.Dense(units=1)
    y_pred = linear_model(x)
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    # create sessions.
    # On windows, currently tensorflow does not allocate all available memory like it says in the documentation,
    # instead you can work around this error by allowing dynamic memory growth as follows:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(init)

    for i in range(100):
        _, loss_value = sess.run((train, loss))
        print(loss_value)

    print(sess.run(y_pred))

    # save the graph
    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())

    sess.close()


if __name__ == "__main__":
    with tf.device('/GPU:0'):
        # test1()
        test_regression()
    pass
