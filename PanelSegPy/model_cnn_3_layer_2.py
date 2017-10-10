"""
3 convolutional layer (use 2 conv2 layer at each layer) NN as base
"""
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def nn_base(img_input=None, trainable=False):

    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv1_1', padding='same')(img_input)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv1_2', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2_1', padding='same')(pool1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2_2', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)
    conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv3_1', padding='same')(pool2)
    conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv3_2', padding='same')(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    return conv3


def nn_classify_label_non_label(img_input=None):
    base_layers = nn_base(img_input, True)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(base_layers)

    flat1 = Flatten(name='flat1')(pool3)
    dense1 = Dense(128, activation='relu', name='dense1')(flat1)
    predictions = Dense(2, activation='softmax', name='output_label_non_label')(dense1)

    return predictions


def rpn(base_layers, num_anchors):

    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]
