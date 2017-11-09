"""
3 convolutional layer NN as base
"""
from keras import backend as K
import tensorflow as tf
from keras.backend import categorical_crossentropy
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, Dropout

from label_rcnn_roi_pooling_conv import RoiPoolingConv


def nn_get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//4

    return get_output_length(width), get_output_length(height)


def nn_base(img_input=None, trainable=False):

    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv1', padding='same')(img_input)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)
    conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv3', padding='same')(pool2)
    # pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    return conv3


def nn_classify_label_non_label(img_input=None):
    base_layers = nn_base(img_input, True)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(base_layers)

    flat1 = Flatten(name='flat1')(pool3)
    dense1 = Dense(128, activation='relu', name='dense1')(flat1)
    predictions = Dense(2, activation='softmax', name='output_label_non_label')(dense1)

    return predictions


def nn_classify_50_plus_bg(img_input=None):
    base_layers = nn_base(img_input, True)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(base_layers)

    flat1 = Flatten(name='flat1')(pool3)
    dense1 = Dense(128, activation='relu', name='dense1')(flat1)
    predictions = Dense(51, activation='softmax', name='output_label_50_plus_bg')(dense1)

    return predictions


lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4


def rpn(base_layers, num_anchors):

    # 3x3 kernel maps to 26x26 pixels in the original image (3->5->10->12->24->26)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes = 51, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    pooling_regions = 7
    input_shape = (num_rois, 7, 7, 512)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]


def nn_rpn(img_input, num_anchors):
    base_layers = nn_base(img_input, True)

    # 3x3 kernel maps to 26x26 pixels in the original image (3->5->10->12->24->26)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def rpn_loss_regr(num_anchors):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
        return lambda_rpn_regr * K.sum(y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])
    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
    return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
        return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
