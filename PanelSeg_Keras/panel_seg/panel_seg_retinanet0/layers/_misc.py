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
from .. import backend
from ..utils import anchors as utils_anchors

import numpy as np


class Anchors(keras.layers.Layer):
    """ Keras layer for generating achors for a given shape.
    """

    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        """ Initializer for an Anchors layer.

        Args
            size: The base size of the anchors to generate.
            stride: The stride of the anchors to generate.
            ratios: The ratios of the anchors to generate (defaults to [0.5, 1, 2]).
            scales: The scales of the anchors to generate (defaults to [2^0, 2^(1/3), 2^(2/3)]).
        """
        self.size   = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            self.ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
        elif isinstance(ratios, list):
            self.ratios  = np.array(ratios)
        if scales is None:
            self.scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
        elif isinstance(scales, list):
            self.scales  = np.array(scales)

        self.num_anchors = len(ratios) * len(scales)
        self.anchors     = keras.backend.variable(utils_anchors.generate_anchors(
            base_size=size,
            ratios=ratios,
            scales=scales,
        ))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = keras.backend.shape(features)[:3]

        # generate proposals from bbox deltas and shifted anchors
        anchors = backend.shift(features_shape[1:3], self.stride, self.anchors)
        anchors = keras.backend.tile(keras.backend.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[1:3]) * self.num_anchors
            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size'   : self.size,
            'stride' : self.stride,
            'ratios' : self.ratios.tolist(),
            'scales' : self.scales.tolist(),
        })

        return config


class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return backend.resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class MergePanelLabel(keras.layers.Layer):
    """
    Keras layer for merging Panel Detection and Label Detection
    Merge pyramids (regression, classification) and l_pyramids (l_regression and l_classification)
    Pyramids: Panel regression (5 tensors with shape (?,?,?,9,4) and classification (5 tensors with shape (?,?,?,9,1):
    At C3, C4, C5, C6, C7 layers
    l_Pyramids: Label regression (4 tensors with shape (?,?,?,9,4) and classification (4 tensors with shape (?,?,?,9,50):
    At C1, C2, C3, C4 layers
    """
    def __init__(self, num_anchors, num_classes, l_num_anchors, l_num_classes, *args, **kwargs):
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.l_num_anchors = l_num_anchors
        self.l_num_classes = l_num_classes
        super(MergePanelLabel, self).__init__(*args, **kwargs)

    def _merge_panel_label(
            self,
            p_r,  # (None, None, None, 9, 4)
            p_c,  # (None, None, None, 9, 1)
            l_r,  # (None, None, None, 9, 4)
            l_c,  # (None, None, None, 9, 50)
            l_c_m,  # (None, None, None, 9)
            scale):

        def a_cond(i, y, x, a, scale,
                   p_r, p_c,
                   l_r, l_c, l_c_m,
                   a_l_r_ta, a_l_c_ta, a_l_c_max_ta):
            shape = keras.backend.shape(p_r)  # (?, ?, ?, 9, 4)
            return keras.backend.less(x, shape[3])

        def a_loop(i, y, x, a, scale,
                   p_r, p_c,
                   l_r, l_c, l_c_m,
                   a_l_r_ta, a_l_c_ta, a_l_c_max_ta):
            l_c_m_roi = l_c_m[i, y * scale:(y + 1) * scale, x * scale:(x + 1) * scale, :]  # l_c_m shape (?, ?, ?, 9)
            l_r_roi = l_r[i, y * scale:(y + 1) * scale, x * scale:(x + 1) * scale, :]  # l_r shape (?, ?, ?, 9, 4)
            l_c_roi = l_c[i, y * scale:(y + 1) * scale, x * scale:(x + 1) * scale, :]  # l_c shape (?, ?, ?, 9, 50)

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
            shape = keras.backend.shape(p_r)  # (?, ?, ?, 9, 4)
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
        l_r_res = keras.backend.reshape(l_r_res, (-1, -1, -1, self.num_anchors, 4))
        l_c_res = keras.backend.reshape(l_c_res, (-1, -1, -1, self.num_anchors, self.l_num_classes))
        l_c_max_res = keras.backend.reshape(l_c_max_res, (-1, -1, -1, self.num_anchors))

        return l_r_res, l_c_res, l_c_max_res

    def merge_panel_label(self, pyramids, l_pyramids):
        # Merge pyramids (regression, classification) and l_pyramids (l_regression and l_classification) here
        # Pyramids: Panel regression (5 tensors with shape (?,?,?,9,4) and classification (5 tensors with shape (?,?,?,9,1):
        # At C3, C4, C5, C6, C7 layers
        # l_Pyramids: Label regression (4 tensors with shape (?,?,?,9,4) and classification (4 tensors with shape (?,?,?,9,50):
        # At C1, C2, C3, C4 layers
        p_regressions = pyramids[0]  # [(?, ?, ?, 9, 4)]*5
        p_classifications = pyramids[1]  # [(?, ?, ?, 9, 1)}*5
        l_regressions = l_pyramids[0]  # [(?, ?, ?, 9, 4)]*4
        l_classifications = l_pyramids[1]  # [(?, ?, ?, 9, 50)]*4

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
                l_r = l_regressions[l_i]  # (?, ?, ?, 9, 4) C1-C4 layers
                l_c = l_classifications[l_i]  # (?, ?, ?, 9, 50) C1-C4 layers
                l_c_m = l_classifications_max[l_i]  # (?, ?, ?, 9) C1-C4 layers
                l_c_layer = l_i + 1

                if p_c_layer - l_c_layer <= 1:
                    continue
                scale = 2 ** (p_c_layer - l_c_layer)

                l_r_res, l_c_res, l_c_m_res = self._merge_panel_label(p_r, p_c, l_r, l_c, l_c_m, scale)

                if l_i == 0:
                    l_r_picked = l_r_res
                    l_c_picked = l_c_res
                    l_c_m_picked = l_c_m_res
                else:
                    cond = keras.backend.greater(l_c_m_res, l_c_m_picked)
                    l_c_m_picked = backend.where(cond, l_c_m_res, l_c_m_picked)
                    # broadcast the cond tensor, such that we could use it for l_regression_res and l_classification_res
                    cond_ex = keras.backend.reshape(cond, shape=[-1, -1, -1, self.l_num_anchors,
                                                                 1])  # convert to [?, ?, ?, ?, 9, 1]
                    # broadcast cond_ex for l_r_picked
                    cond_bc1 = keras.backend.tile(cond_ex,
                                                  keras.backend.stack([1, 1, 1, 1, keras.backend.shape(l_r_res)[-1]]))
                    l_r_picked = backend.where(cond_bc1, l_r_res, l_r_picked)
                    # broadcast cond_ex for l_c_picked
                    cond_bc2 = keras.backend.tile(cond_ex,
                                                  keras.backend.stack([1, 1, 1, 1, keras.backend.shape(l_c_res)[-1]]))
                    l_c_picked = backend.where(cond_bc2, l_c_res, l_c_picked)

            # reshape
            p_r_reshape = keras.layers.Reshape((-1, 4), name='p_regression_reshape')(p_r)
            p_c_reshape = keras.layers.Reshape((-1, self.num_classes), name='p_classification_reshape')(p_c)
            l_r_reshape = keras.layers.Reshape((-1, 4), name='l_regression_reshape')(l_r_picked)
            l_c_reshape = keras.layers.Reshape((-1, self.l_num_classes), name='l_classification_reshape')(l_c_picked)

            p_r_outputs.append(p_r_reshape)
            p_c_outputs.append(p_c_reshape)
            l_r_outputs.append(l_r_reshape)
            l_c_outputs.append(l_c_reshape)

        p_r_outputs = keras.layers.Concatenate(axis=1, name='p_regression')(p_r_outputs)
        p_c_outputs = keras.layers.Concatenate(axis=1, name='p_classification')(p_c_outputs)
        l_r_outputs = keras.layers.Concatenate(axis=1, name='l_regression')(l_r_outputs)
        l_c_outputs = keras.layers.Concatenate(axis=1, name='l_classification')(l_c_outputs)

        return [
            p_r_outputs,
            p_c_outputs,
            l_r_outputs,
            l_c_outputs
        ]

    def call(self, inputs, **kwargs):
        p_regressions = inputs[:5]  # [(?, ?, ?, 9, 4)]*5
        p_classifications = inputs[5:10]  # [(?, ?, ?, 9, 1)}*5
        l_regressions = inputs[10:14]  # [(?, ?, ?, 9, 4)]*4
        l_classifications = inputs[14:]  # [(?, ?, ?, 9, 50)]*4

        pyramids = [p_regressions, p_classifications]
        l_pyramids = [l_regressions, l_classifications]
        outputs = self.merge_panel_label(pyramids, l_pyramids)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = [(input_shape[0][0], None, 4)] * 5 + \
                       [(input_shape[0][0], None, self.num_classes)] * 5 + \
                       [(input_shape[0][0], None, 4)] * 5 + \
                       [(input_shape[0][0], None, self.l_num_classes)] * 5
        return output_shape


class RegressBoxes(keras.layers.Layer):
    """ Keras layer for applying regression values to boxes.
    """

    def __init__(self, mean=None, std=None, *args, **kwargs):
        """ Initializer for the RegressBoxes layer.

        Args
            mean: The mean value of the regression values which was used for normalization.
            std: The standard value of the regression values which was used for normalization.
        """
        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.2, 0.2, 0.2, 0.2])

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        self.mean = mean
        self.std  = std
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return backend.bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std' : self.std.tolist(),
        })

        return config


class ClipBoxes(keras.layers.Layer):
    """ Keras layer to clip box values to lie inside a given shape.
    """

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())

        x1 = backend.clip_by_value(boxes[:, :, 0], 0, shape[2])
        y1 = backend.clip_by_value(boxes[:, :, 1], 0, shape[1])
        x2 = backend.clip_by_value(boxes[:, :, 2], 0, shape[2])
        y2 = backend.clip_by_value(boxes[:, :, 3], 0, shape[1])

        return keras.backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]
