"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras

from panel_seg.panel_seg_retinanet.layers import MergePanelLabel
from .. import initializers
from .. import layers
from .. import backend

import numpy as np


def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    """ Creates the default classification submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    # outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Reshape((-1, -1, num_anchors, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    """ Creates the default regression submodel.

    Args
        num_anchors             : Number of anchors to regress for each feature level.
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    # outputs = keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)
    outputs = keras.layers.Reshape((-1, -1, num_anchors, 4), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_l_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=64,
    prior_probability=0.01,
    classification_feature_size=64,
    name='l_classification_submodel'
):
    """ Creates the default label classification submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_l_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.zeros(),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_l_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    # outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_l_classification_reshape')(outputs)
    outputs = keras.layers.Reshape((-1, -1, num_anchors, num_classes), name='pyramid_l_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_l_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_l_regression_model(num_anchors, pyramid_feature_size=64, regression_feature_size=64, name='l_regression_submodel'):
    """ Creates the default regression submodel.

    Args
        num_anchors             : Number of anchors to regress for each feature level.
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_l_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * 4, name='pyramid_l_regression', **options)(outputs)
    # outputs = keras.layers.Reshape((-1, 4), name='pyramid_l_regression_reshape')(outputs)
    outputs = keras.layers.Reshape((-1, -1, num_anchors, 4), name='pyramid_l_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]


def __create_l_pyramid_features(C1, C2, C3, C4, feature_size=64):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C1           : Feature stage C1 from the backbone.
        C2           : Feature stage C2 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P1, P2].
    """

    # upsample C4 to get P4 from the FPN paper
    L_P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='L_C4_reduced')(C4)
    L_P4_upsampled = layers.UpsampleLike(name='L_P4_upsampled')([L_P4, C3])
    L_P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='L_P4')(L_P4)

    # add P4 elementwise to C3
    L_P3           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='L_C3_reduced')(C3)
    L_P3           = keras.layers.Add(name='L_P3_merged')([L_P4_upsampled, L_P3])
    L_P3_upsampled = layers.UpsampleLike(name='L_P3_upsampled')([L_P3, C2])
    L_P3           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='L_P3')(L_P3)

    # add P3 elementwise to C2
    L_P2           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='L_C2_reduced')(C2)
    L_P2           = keras.layers.Add(name='L_P2_merged')([L_P3_upsampled, L_P2])
    L_P2_upsampled = layers.UpsampleLike(name='L_P2_upsampled')([L_P2, C1])
    L_P2           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='L_P2')(L_P2)

    # add P2 elementwise to C1
    L_P1 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='L_C1_reduced')(C1)
    L_P1 = keras.layers.Add(name='L_P1_merged')([L_P2_upsampled, L_P1])
    L_P1 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='L_P1')(L_P1)

    return [L_P1, L_P2, L_P3, L_P4]


class AnchorParameters:
    """ The parameteres that define how anchors are generated.

    Args
        sizes   : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios  : List of ratios to use per location in a feature map.
        scales  : List of scales to use per location in a feature map.
    """
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


"""
The default anchor parameters.
"""
AnchorParameters.panel = AnchorParameters(
    sizes   = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
    scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)


"""
The default label anchor parameters.
"""
AnchorParameters.label = AnchorParameters(
    sizes   = [8, 16, 32, 64],
    strides = [2, 4, 8, 16],
    ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
    scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)


def default_submodels(num_classes, num_anchors):
    """ Create a list of default submodels used for panel detection.

    The default submodels contains a regression submodel and a classification submodel.

    Args
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.

    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    return [
        ('regression', default_regression_model(num_anchors)),
        ('classification', default_classification_model(num_classes, num_anchors)),
    ]


def default_l_submodels(l_num_classes, l_num_anchors):
    """ Create a list of default submodels used for label detection.

    The default submodels contains a regression submodel and a classification submodel.

    Args
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.

    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    return [
        ('l_regression', default_l_regression_model(l_num_anchors)),
        ('l_classification', default_l_classification_model(l_num_classes, l_num_anchors))
    ]


def __build_model_pyramid(name, model, features):
    """ Applies a single submodel to each FPN level.

    Args
        name     : Name of the submodel.
        model    : The submodel to evaluate.
        features : The FPN features.

    Returns
        A tensor containing the response from the submodel on the FPN features.
    """
    # return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])
    return [model(f) for f in features]


def __build_pyramid(models, features):
    """ Applies all submodels to each FPN level.

    Args
        models   : List of sumodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.

    Returns
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def __l_build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='l_anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='l_anchors')(anchors)


def _merge_panel_label(
        p_r,    # (None, None, None, 9, 4)
        p_c,    # (None, None, None, 9, 1)
        l_r,    # (None, None, None, 9, 4)
        l_c,    # (None, None, None, 9, 50)
        l_c_m,  # (None, None, None, 9)
        scale, num_anchors, l_num_anchors, l_num_classes):

    def a_cond(i, y, x, a, scale,
               p_r, p_c,
               l_r, l_c, l_c_m,
               a_l_r_ta, a_l_c_ta, a_l_c_max_ta):
        shape = keras.backend.shape(p_r)    # (?, ?, ?, 9, 4)
        return keras.backend.less(x, shape[3])

    def a_loop(i, y, x, a, scale,
               p_r, p_c,
               l_r, l_c, l_c_m,
               a_l_r_ta, a_l_c_ta, a_l_c_max_ta):

        l_c_m_roi = l_c_m[i, y*scale:(y+1)*scale, x*scale:(x+1)*scale, :]  #l_c_m shape (?, ?, ?, 9)
        l_r_roi = l_r[i, y*scale:(y+1)*scale, x*scale:(x+1)*scale, :]      #l_r shape (?, ?, ?, 9, 4)
        l_c_roi = l_c[i, y*scale:(y+1)*scale, x*scale:(x+1)*scale, :]      #l_c shape (?, ?, ?, 9, 50)

        l_c_m_roi_flat = keras.backend.reshape(l_c_m_roi, [-1])
        l_r_roi_flat = keras.backend.reshape(l_r_roi, [-1, 4])
        l_c_roi_flat = keras.backend.reshape(l_c_roi, [-1, 50])

        ind_max = keras.backend.argmax(l_c_m_roi_flat)
        l_c_max_max = keras.backend.gather(l_c_m_roi_flat, ind_max)
        l_r_max = keras.backend.gather(l_r_roi_flat, ind_max)
        l_c_max = keras.backend.gather(l_c_roi_flat, ind_max)

        a_l_r_ta.write(a, l_r_max)
        a_l_c_ta.write(a, l_c_max)
        a_l_c_max_ta.write(a, l_c_max_max)

        return [i, y, x, backend.add(a, 1), scale,
                p_r, p_c,
                l_r, l_c, l_c_m,
                a_l_r_ta, a_l_c_ta, a_l_c_max_ta]

    def x_cond(i, y, x, scale,
               p_r, p_c,
               l_r, l_c, l_c_m,
               x_l_r_ta, x_l_c_ta, x_l_c_max_ta):
        shape = keras.backend.shape(p_r)    # (?, ?, ?, 9, 4)
        return keras.backend.less(x, shape[2])

    def x_loop(i, y, x, scale,
               p_r, p_c,
               l_r, l_c, l_c_m,
               x_l_r_ta, x_l_c_ta, x_l_c_max_ta):

        a_dtype = keras.backend.dtype(p_r)
        a_size = keras.backend.shape(p_r)[3]
        a_l_r_ta = backend.create_tensorarray(dtype=a_dtype, size=a_size)
        a_l_c_ta = backend.create_tensorarray(dtype=a_dtype, size=a_size)
        a_l_c_max_ta = backend.create_tensorarray(dtype=a_dtype, size=a_size)

        _, _, _, _, _, _, _, _, _, _, a_l_r_ta, a_l_c_ta, a_l_c_max_ta \
            = backend.while_loop(a_cond, a_loop,
                                 [
                                     i, y, x, 0, scale,
                                     p_r, p_c,
                                     l_r, l_c, l_c_m,
                                     a_l_r_ta, a_l_c_ta, a_l_c_max_ta
                                 ])
        a_l_r_res = a_l_r_ta.stack()
        a_l_c_res = a_l_c_ta.stack()
        a_l_c_max_res = a_l_c_max_ta.stack()

        x_l_r_ta.write(x, a_l_r_res)
        x_l_c_ta.write(x, a_l_c_res)
        x_l_c_max_ta.write(x, a_l_c_max_res)

        return [i, y, backend.add(x, 1), scale,
                p_r, p_c,
                l_r, l_c, l_c_m,
                x_l_r_ta, x_l_c_ta, x_l_c_max_ta]

    def y_cond(i, y, scale,
               p_r, p_c,
               l_r, l_c, l_c_m,
               y_l_r_ta, y_l_c_ta, y_l_c_max_ta
               ):
        shape = keras.backend.shape(p_r)
        return keras.backend.less(y, shape[1])

    def y_loop(i, y, scale,
               p_r, p_c,
               l_r, l_c, l_c_m,
               y_l_r_ta, y_l_c_ta, y_l_c_max_ta):

        x_dtype = keras.backend.dtype(p_r)
        x_size = keras.backend.shape(p_r)[2]
        x_l_r_ta = backend.create_tensorarray(dtype=x_dtype, size=x_size)
        x_l_c_ta = backend.create_tensorarray(dtype=x_dtype, size=x_size)
        x_l_c_max_ta = backend.create_tensorarray(dtype=x_dtype, size=x_size)

        _, _, _, _, _, _, _, _, _, x_l_r_ta, x_l_c_ta, x_l_c_max_ta \
            = backend.while_loop(x_cond, x_loop,
                                 [
                                     i, y, 0, scale,
                                     p_r, p_c,
                                     l_r, l_c, l_c_m,
                                     x_l_r_ta, x_l_c_ta, x_l_c_max_ta
                                 ])

        x_l_r_res = x_l_r_ta.stack()
        x_l_c_res = x_l_c_ta.stack()
        x_l_c_max_res = x_l_c_max_ta.stack()

        y_l_r_ta.write(y, x_l_r_res)
        y_l_c_ta.write(y, x_l_c_res)
        y_l_c_max_ta.write(y, x_l_c_max_res)

        return [i, backend.add(y, 1), scale,
                p_r, p_c,
                l_r, l_c, l_c_m,
                y_l_r_ta, y_l_c_ta, y_l_c_max_ta]

    def batch_cond(i, scale,
                   p_r, p_c,
                   l_r, l_c, l_c_m,
                   i_l_r_ta, i_l_c_ta, i_l_c_max_ta):
        shape = keras.backend.shape(p_r)
        return keras.backend.less(i, shape[0])

    def batch_loop(i, scale,
                   p_r, p_c,
                   l_r, l_c, l_c_m,
                   i_l_r_ta, i_l_c_ta, i_l_c_max_ta):

        y_dtype = keras.backend.dtype(p_r)
        y_size = keras.backend.shape(p_r)[1]
        y_l_r_ta = backend.create_tensorarray(dtype=y_dtype, size=y_size)
        y_l_c_ta = backend.create_tensorarray(dtype=y_dtype, size=y_size)
        y_l_c_max_ta = backend.create_tensorarray(dtype=y_dtype, size=y_size)

        _, _, _, _, _, _, _, _, y_l_r_ta, y_l_c_ta, y_l_c_max_ta \
            = backend.while_loop(y_cond, y_loop,
                                 [
                                     i, 0, scale,
                                     p_r, p_c,
                                     l_r, l_c, l_c_m,
                                     y_l_r_ta, y_l_c_ta, y_l_c_max_ta
                                 ])

        y_l_r_res = y_l_r_ta.stack()
        y_l_c_res = y_l_c_ta.stack()
        y_l_c_max_res = y_l_c_max_ta.stack()

        i_l_r_ta.write(i, y_l_r_res)
        i_l_c_ta.write(i, y_l_c_res)
        i_l_c_max_ta.write(i, y_l_c_max_res)

        return [backend.add(i, 1), scale,
                p_r, p_c,
                l_r, l_c, l_c_m,
                i_l_r_ta, i_l_c_ta, i_l_c_max_ta
                ]

    i_dtype = keras.backend.dtype(p_r)
    i_size = keras.backend.shape(p_r)[0]
    i_l_r_ta = backend.create_tensorarray(dtype=i_dtype, size=i_size)
    i_l_c_ta = backend.create_tensorarray(dtype=i_dtype, size=i_size)
    i_l_c_max_ta = backend.create_tensorarray(dtype=i_dtype, size=i_size)

    _, _, _, _, _, _, _, i_l_r_ta, i_l_c_ta, i_l_c_max_ta \
        = backend.while_loop(batch_cond, batch_loop,
                             [0, scale,
                              p_r, p_c,
                              l_r, l_c, l_c_m,
                              i_l_r_ta, i_l_c_ta, i_l_c_max_ta])
    l_r_res = i_l_r_ta.stack()
    l_c_res = i_l_c_ta.stack()
    l_c_max_res = i_l_c_max_ta.stack()

    # we know the last 2 dimensions, so we reshape it.
    l_r_res = keras.backend.reshape(l_r_res, (-1, -1, -1, num_anchors, 4))
    l_c_res = keras.backend.reshape(l_c_res, (-1, -1, -1, num_anchors, l_num_classes))
    l_c_max_res = keras.backend.reshape(l_c_max_res, (-1, -1, -1, num_anchors))

    return l_r_res, l_c_res, l_c_max_res


def merge_panel_label(p_regressions, p_classifications, l_regressions, l_classifications,
                      num_anchors, num_classes, l_num_anchors, l_num_classes):
    # Merge pyramids (regression, classification) and l_pyramids (l_regression and l_classification) here
    # Pyramids: Panel regression (5 tensors with shape (?,?,?,9,4) and classification (5 tensors with shape (?,?,?,9,1):
    # At C3, C4, C5, C6, C7 layers
    # l_Pyramids: Label regression (4 tensors with shape (?,?,?,9,4) and classification (4 tensors with shape (?,?,?,9,50):
    # At C1, C2, C3, C4 layers

    l_classifications_max = [keras.backend.max(c, axis=-1) for c in l_classifications]  # [(?, ?, ?, 9)]*4

    p_r_outputs = []
    p_c_outputs = []
    l_r_outputs = []
    l_c_outputs = []

    for p_i in range(len(p_regressions)):
        p_r = p_regressions[p_i]  # (?, ?, ?, 9, 4) C3-C7 layers
        p_c = p_classifications[p_i]  # (?, ?, ?, 9, 1) C3-C7 layers

        p_c_layer = p_i + 3

        l_r_picked = None
        l_c_picked = None
        l_c_m_picked = None

        for l_i in range(len(l_regressions)):
            l_r = l_regressions[l_i]                   # (?, ?, ?, 9, 4) C1-C4 layers
            l_c = l_classifications[l_i]           # (?, ?, ?, 9, 50) C1-C4 layers
            l_c_m = l_classifications_max[l_i]   # (?, ?, ?, 9) C1-C4 layers
            l_c_layer = l_i + 1

            if p_c_layer - l_c_layer <= 1:
                continue
            scale = 2 ** (p_c_layer - l_c_layer)

            l_r_res, l_c_res, l_c_m_res = _merge_panel_label(p_r, p_c, l_r, l_c, l_c_m,scale, num_anchors, l_num_anchors, l_num_classes)

            if l_i == 0:
                l_r_picked = l_r_res
                l_c_picked = l_c_res
                l_c_m_picked = l_c_m_res
            else:
                cond = keras.backend.greater(l_c_m_res, l_c_m_picked)
                l_c_m_picked = backend.where(cond, l_c_m_res, l_c_m_picked)
                # broadcast the cond tensor, such that we could use it for l_regression_res and l_classification_res
                cond_ex = keras.backend.reshape(cond, shape=[-1, -1, -1, l_num_anchors, 1])    # convert to [?, ?, ?, ?, 9, 1]
                # broadcast cond_ex for l_r_picked
                cond_bc1 = keras.backend.tile(cond_ex, keras.backend.stack([1,1,1,1,keras.backend.shape(l_r_res)[-1]]))
                l_r_picked = backend.where(cond_bc1, l_r_res, l_r_picked)
                # broadcast cond_ex for l_c_picked
                cond_bc2 = keras.backend.tile(cond_ex, keras.backend.stack([1,1,1,1,keras.backend.shape(l_c_res)[-1]]))
                l_c_picked = backend.where(cond_bc2, l_c_res, l_c_picked)

        # reshape
        p_r_reshape = keras.layers.Reshape((-1, 4), name='p_regression_reshape')(p_r)
        p_c_reshape = keras.layers.Reshape((-1, num_classes), name='p_classification_reshape')(p_c)
        l_r_reshape = keras.layers.Reshape((-1, 4), name='l_regression_reshape')(l_r_picked)
        l_c_reshape = keras.layers.Reshape((-1, l_num_classes), name='l_classification_reshape')(l_c_picked)

        p_r_outputs.append(p_r_reshape)
        p_c_outputs.append(p_c_reshape)
        l_r_outputs.append(l_r_reshape)
        l_c_outputs.append(l_c_reshape)

    return [
        p_r_outputs,
        p_c_outputs,
        l_r_outputs,
        l_c_outputs
    ]


def merge_panel_label_lambda(args):
    p_regressions = args[:5]            # [(?, ?, ?, 9, 4)]*5
    p_classifications = args[5:10]      # [(?, ?, ?, 9, 1)}*5
    l_regressions = args[10:14]         # [(?, ?, ?, 9, 4)]*4
    l_classifications = args[14:]       # [(?, ?, ?, 9, 50)]*4

    num_anchors = keras.backend.shape(p_classifications[0])[3]
    num_classes = keras.backend.shape(p_classifications[0])[4]
    l_num_anchors = keras.backend.shape(l_classifications[0])[3]
    l_num_classes = keras.backend.shape(l_classifications[0])[4]

    return merge_panel_label(p_regressions, p_classifications, l_regressions, l_classifications,
                             num_anchors, num_classes, l_num_anchors, l_num_classes)


def merge_panel_label_max_pooling(
        p_regressions,      # [(?, ?, ?, 9, 4)]*5
        p_classifications,  # [(?, ?, ?, 9, 1)]*5
        l_regressions,      # [(?, ?, ?, 9, 4)]*4
        l_classifications,   # [(?, ?, ?, 9, 50)]*4
        num_anchors, num_classes, l_num_anchors, l_num_classes
):
    p_r_outputs = []
    p_c_outputs = []
    l_c_outputs = []

    for p_i in range(len(p_regressions)):
        p_r = p_regressions[p_i]        # (?, ?, ?, 9, 4) C3-C7 layers
        p_c = p_classifications[p_i]    # (?, ?, ?, 9, 1) C3-C7 layers

        p_c_layer = p_i + 3

        l_c_max_all = []
        for l_i in range(len(l_regressions)):
            l_r = l_regressions[l_i]  # (?, ?, ?, 9, 4) C1-C4 layers
            l_c = l_classifications[l_i]  # (?, ?, ?, 9, 50) C1-C4 layers
            l_c_layer = l_i + 1

            if p_c_layer - l_c_layer <= 1:
                continue
            scale = 2 ** (p_c_layer - l_c_layer)

            l_c_max = keras.layers.MaxPooling3D((scale, scale, l_num_anchors), strides=(scale,scale,l_num_anchors), padding='same')(l_c)
            l_c_max_all.append(l_c_max)

        if len(l_c_max_all) == 1:
            l_c = l_c_max_all[0]
        else:
            l_c_max = keras.layers.Concatenate(axis=3)(l_c_max_all)
            l_c = keras.layers.MaxPooling3D((1, 1, len(l_c_max_all)), strides=(1, 1, len(l_c_max_all)),
                                            padding='same')(l_c_max)

        l_c_list = [l_c] * num_anchors
        l_c = keras.layers.Concatenate(axis=3)(l_c_list)

        p_r_reshaped = keras.layers.Reshape((-1, 4))(p_r)
        p_c_reshaped = keras.layers.Reshape((-1, num_classes))(p_c)
        l_c_reshaped = keras.layers.Reshape((-1, l_num_classes))(l_c)

        p_r_outputs.append(p_r_reshaped)
        p_c_outputs.append(p_c_reshaped)
        l_c_outputs.append(l_c_reshaped)

    p_r_outputs = keras.layers.Concatenate(axis=1, name='p_regression')(p_r_outputs)
    p_c_outputs = keras.layers.Concatenate(axis=1, name='p_classification')(p_c_outputs)
    l_c_outputs = keras.layers.Concatenate(axis=1, name='l_classification')(l_c_outputs)

    return p_r_outputs, p_c_outputs, l_c_outputs


def retinanet(
    inputs,
    backbone_layers,
    num_classes,
    l_num_classes,
    num_anchors             = 9,
    l_num_anchors           = 9,
    create_pyramid_features = __create_pyramid_features,
    create_l_pyramid_features = __create_l_pyramid_features,
    submodels               = None,
    name                    = 'retinanet'
):
    """ Construct a RetinaNet model on top of a backbone.

    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).

    Args
        inputs                  : keras.layers.Input (or list of) for the input to the model.
        num_classes             : Number of classes to classify.
        num_anchors             : Number of base anchors.
        create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
        submodels               : Submodels to run on each feature map (default is regression and classification submodels).
        name                    : Name of the model.

    Returns
        A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.

        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, l_regression, l_classification, other[0], other[1], ...
        ]
        ```
    """
    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors)
        l_submodels = default_l_submodels(l_num_classes, l_num_anchors)

    C1, C2, C3, C4, C5 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = create_pyramid_features(C3, C4, C5)
    l_features = create_l_pyramid_features(C1, C2, C3, C4)

    # for all pyramid levels, run available submodels
    pyramids = __build_pyramid(submodels, features)
    l_pyramids = __build_pyramid(l_submodels, l_features)

    p_regressions = pyramids[0]         # [(?, ?, ?, 9, 4)]*5
    p_classifications = pyramids[1]     # [(?, ?, ?, 9, 1)}*5
    l_regressions = l_pyramids[0]       # [(?, ?, ?, 9, 4)]*4
    l_classifications = l_pyramids[1]   # [(?, ?, ?, 9, 50)]*4

    outputs = merge_panel_label_max_pooling(p_regressions, p_classifications, l_regressions, l_classifications,
                                            num_anchors, num_classes, l_num_anchors, l_num_classes)

    # merge_inputs = p_regressions + p_classifications + l_regressions + l_classifications
    # outputs = MergePanelLabel(num_anchors, num_classes, l_num_anchors, l_num_classes, trainable=False)(merge_inputs)
    # outputs = keras.layers.Lambda(merge_panel_label_lambda)(merge_inputs)

    # outputs = merge_panel_label(p_regressions, p_classifications, l_regressions, l_classifications,
    #                             num_anchors, num_classes, l_num_anchors, l_num_classes)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def retinanet_bbox(
    model             = None,
    panel_anchor_parameters = AnchorParameters.panel,
    label_anchor_parameters = AnchorParameters.label,
    nms               = True,
    name              = 'retinanet-bbox',
    **kwargs
):
    """ Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.

    This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.

    Args
        model             : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        anchor_parameters : Struct containing configuration for anchor generation (sizes, strides, ratios, scales).
        name              : Name of the model.
        *kwargs           : Additional kwargs to pass to the minimal retinanet model.

    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """
    if model is None:
        model = retinanet(num_anchors=panel_anchor_parameters.num_anchors(), l_num_anchors=label_anchor_parameters.num_anchors(), **kwargs)

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    anchors  = __build_anchors(panel_anchor_parameters, features)

    # compute the label anchors
    l_features = [model.get_layer(p_name).output for p_name in ['L_P1', 'L_P2', 'L_P3', 'L_P4']]
    # l_anchors  = __l_build_anchors(label_anchor_parameters, l_features)

    # we expect the anchors, regression and classification values as first output
    regression     = model.outputs[0]
    classification = model.outputs[1]
    # l_regression     = model.outputs[2]
    l_classification = model.outputs[2]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[3:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # apply predicted regression to label anchors
    # l_boxes = layers.RegressBoxes(name='l_boxes')([l_anchors, l_regression])
    # l_boxes = layers.ClipBoxes(name='l_clipped_boxes')([model.inputs[0], l_boxes])

    # filter detections (apply NMS / score threshold / merge panel and label candidates / select top-k)
    detections = layers.FilterDetections(
        nms=nms,
        name='filtered_detections'
    )([boxes, classification, l_classification] + other)

    outputs = detections

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=outputs, name=name)
