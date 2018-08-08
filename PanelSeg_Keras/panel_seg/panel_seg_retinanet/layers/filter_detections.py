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


def filter_detections(boxes, classification, l_classification, other=[], nms=True, score_threshold=0.05, max_detections=300, nms_threshold=0.5):
    """ Filter detections using the boxes and classification values.

    Args
        boxes           : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification  : Tensor of shape (num_boxes, num_classes) containing the classification scores.
        other           : List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
        nms             : Flag to enable/disable non maximum suppression.
        score_threshold : Threshold used to prefilter the boxes with.
        max_detections  : Maximum number of detections to keep.
        nms_threshold   : Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """
    all_indices = []

    # perform per class filtering
    for c in range(int(classification.shape[1])):
        scores = classification[:, c]

        # threshold based on score
        indices = backend.where(keras.backend.greater(scores, score_threshold))

        if nms:
            filtered_boxes  = backend.gather_nd(boxes, indices)
            filtered_scores = keras.backend.gather(scores, indices)[:, 0]

            # perform NMS
            nms_indices = backend.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)

            # filter indices based on NMS
            indices = keras.backend.gather(indices, nms_indices)

        # add indices to list of all indices
        labels  = c * keras.backend.ones((keras.backend.shape(indices)[0],), dtype='int64')
        indices = keras.backend.stack([indices[:, 0], labels], axis=1)
        all_indices.append(indices)

    # concatenate indices to single tensor
    indices = keras.backend.concatenate(all_indices, axis=0)

    # select top k
    scores              = backend.gather_nd(classification, indices)
    labels              = indices[:, 1]
    scores, top_indices = backend.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices             = keras.backend.gather(indices[:, 0], top_indices)
    boxes               = keras.backend.gather(boxes, indices)
    labels              = keras.backend.gather(labels, top_indices)

    # l_boxes             = keras.backend.gather(l_boxes, indices)
    l_classification    = keras.backend.gather(l_classification, indices)
    l_labels = keras.backend.argmax(l_classification, axis=-1)
    l_scores = keras.backend.max(l_classification, axis=-1)

    other_              = [keras.backend.gather(o, indices) for o in other]

    labels   = keras.backend.cast(labels, 'int32')
    l_labels = keras.backend.cast(l_labels, 'int32')

    # Find l_scores and l_labels

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes    = backend.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores   = backend.pad(scores, [[0, pad_size]], constant_values=-1)
    labels   = backend.pad(labels, [[0, pad_size]], constant_values=-1)
    labels   = keras.backend.cast(labels, 'int32')

    # l_boxes    = backend.pad(l_boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    l_scores   = backend.pad(l_scores, [[0, pad_size]], constant_values=-1)
    l_labels   = backend.pad(l_labels, [[0, pad_size]], constant_values=-1)
    l_labels   = keras.backend.cast(l_labels, 'int32')

    other_   = [backend.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    # l_boxes.set_shape([max_detections, 4])
    l_scores.set_shape([max_detections])
    l_labels.set_shape([max_detections])
    for o, s in zip(other_, [list(keras.backend.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels, l_scores, l_labels] + other_


def merge_panel_label_while_loop(p_boxes, p_scores, p_labels, l_boxes, l_scores, l_labels, max_detections):

    num_p_boxes = keras.backend.shape(p_boxes)[0]
    num_l_boxes = keras.backend.shape(l_boxes)[0]

    def cond_p_boxes(i, pboxes):
        return keras.backend.less(i, num_p_boxes)

    def loop_p_boxes(i, pboxes):
        p_box = pboxes[i]
        backend.while_loop(cond_l_boxes, body_l_boxes, [0, p_box, l_boxes])
        return i+1

    def cond_l_boxes(i, pbox, lboxes):
        l_box = lboxes[i]
        pass

    def body_l_boxes(i, pbox, lboxes):
        l_box = lboxes[i]
        return i+1

    backend.while_loop(cond_p_boxes, loop_p_boxes, [0, p_boxes])


def merge_panel_label_map_fn(p_boxes, p_scores, p_labels, l_boxes, l_scores, l_labels, max_detections):
    """ Merge Panel and Label result.
    We pick the label candidate having largest score in the p_boxes

    Args
        p_boxes           : Tensor of shape (num_boxes, 4) containing the panel boxes in (x1, y1, x2, y2) format.
        p_scores          : Tensor of shape (num_boxes, num_classes) containing the panel classification scores.
        p_classification  : Tensor of shape (num_boxes, num_classes) containing the panel classification labels.
        l_boxes         : Tensor of shape (num_boxes, 4) containing the label boxes in (x1, y1, x2, y2) format.
        l_scores        : Tensor of shape (num_boxes, num_classes) containing the label classification scores.
        classification: Tensor of shape (num_boxes, num_classes) containing the label classification labels.
        max_detections  : Maximum number of detections to keep.

    Returns

        A list of [boxes, scores, labels, l_boxes, l_scores, l_labels].
        p_boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed panel boxes.
        p_scores is shaped (max_detections,) and contains the scores of the predicted panel class.
        p_labels is shaped (max_detections,) and contains the predicted panel label.
        l_boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed label boxes.
        l_scores is shaped (max_detections,) and contains the scores of the predicted label class.
        l_labels is shaped (max_detections,) and contains the predicted label label.

        In case there are less than max_detections detections, the tensors are padded with -1's.
        In case there is no label candidates for p_boxes, the label tensors are padded with -1's.
    """

    keras.backend.int_shape(l_boxes)[0]

    def _merge_panel_label(p_box):
        p_x1 = p_box[0]
        p_y1 = p_box[1]
        p_x2 = p_box[2]
        p_y2 = p_box[3]

        l_x1 = l_boxes[:, 0]
        l_y1 = l_boxes[:, 1]
        l_x2 = l_boxes[:, 2]
        l_y2 = l_boxes[:, 3]

        condition1 = keras.backend.greater_equal(l_x1, p_x1)  # l_x1 >= p_x1
        condition2 = keras.backend.greater(l_x2, p_x1)        # l_x2 >= p_x1
        condition3 = keras.backend.greater_equal(l_y1, p_y1)  # l_y1 >= p_y1
        condition4 = keras.backend.greater(l_y2, p_y1)        # l_y2 >= p_y1
        condition5 = keras.backend.less(l_x1, p_x2)           # l_x1 <= p_x2
        condition6 = keras.backend.less_equal(l_x2, p_x2)     # l_x2 <= p_x2
        condition7 = keras.backend.less(l_y1, p_y2)           # l_y1 <= p_y2
        condition8 = keras.backend.less_equal(l_y2, p_y2)     # l_y2 <= p_y2

        inside_flag = condition1
        inside_flag = backend.logical_and(inside_flag, condition2)
        inside_flag = backend.logical_and(inside_flag, condition3)
        inside_flag = backend.logical_and(inside_flag, condition4)
        inside_flag = backend.logical_and(inside_flag, condition5)
        inside_flag = backend.logical_and(inside_flag, condition6)
        inside_flag = backend.logical_and(inside_flag, condition7)
        inside_flag = backend.logical_and(inside_flag, condition8)

        indices = backend.where(inside_flag)

        return indices  # we just pick the top one

    inside_indices = backend.map_fn(
        _merge_panel_label,
        elems=p_boxes,
        dtype='int64'
    )

    # collect label candidate based on indices
    l_boxes = keras.backend.gather(l_boxes, inside_indices[:, 0, 0])
    l_scores = keras.backend.gather(l_scores, inside_indices[:, 0, 0])
    l_labels = keras.backend.gather(l_labels, inside_indices[:, 0, 0])

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(p_scores)[0])
    p_boxes    = backend.pad(p_boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    p_scores   = backend.pad(p_scores, [[0, pad_size]], constant_values=-1)
    p_labels   = backend.pad(p_labels, [[0, pad_size]], constant_values=-1)
    p_labels   = keras.backend.cast(p_labels, 'int32')
    l_boxes    = backend.pad(l_boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    l_scores   = backend.pad(l_scores, [[0, pad_size]], constant_values=-1)
    l_labels   = backend.pad(l_labels, [[0, pad_size]], constant_values=-1)
    l_labels   = keras.backend.cast(l_labels, 'int32')

    # set shapes, since we know what they are
    p_boxes.set_shape([max_detections, 4])
    p_scores.set_shape([max_detections])
    p_labels.set_shape([max_detections])
    l_boxes.set_shape([max_detections, 4])
    l_scores.set_shape([max_detections])
    l_labels.set_shape([max_detections])

    return [p_boxes, p_scores, p_labels, l_boxes, l_scores, l_labels]


class FilterDetections(keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
        self,
        nms                 = True,
        nms_threshold       = 0.5,
        score_threshold     = 0.05,
        max_detections      = 300,
        l_nms               = True,
        l_nms_threshold     = 0.5,
        l_score_threshold   = 0.05,
        l_max_detections    = 300,
        merge_max_detections= 30,
        parallel_iterations = 32,
        **kwargs
    ):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            nms                 : Flag to enable/disable NMS.
            nms_threshold       : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold     : Threshold used to prefilter the boxes with.
            max_detections      : Maximum number of detections to keep.
            parallel_iterations : Number of batch items to process in parallel.
        """
        self.nms                 = nms
        self.nms_threshold       = nms_threshold
        self.score_threshold     = score_threshold
        self.max_detections      = max_detections
        self.l_nms               = l_nms
        self.l_nms_threshold     = l_nms_threshold
        self.l_score_threshold   = l_score_threshold
        self.l_max_detections    = l_max_detections
        self.merge_max_detections= merge_max_detections
        self.parallel_iterations = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes          = inputs[0]
        classification = inputs[1]
        # l_boxes          = inputs[2]
        l_classification = inputs[2]
        other          = inputs[3:]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes          = args[0]
            classification = args[1]
            # l_boxes = args[2]
            l_classification = args[2]
            other = args[3]

            return filter_detections(
                boxes,
                classification,
                # l_boxes,
                l_classification,
                other,
                nms=self.nms,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold,
            )

        outputs = backend.map_fn(
            _filter_detections,
            elems=[boxes, classification, l_classification, other],
            dtype=[keras.backend.floatx(), keras.backend.floatx(), 'int32', keras.backend.floatx(), 'int32'] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )

        # call filter_detections on each batch
        # p_outputs = backend.map_fn(
        #     _filter_detections,
        #     elems=[boxes, classification, other],
        #     dtype=[keras.backend.floatx(), keras.backend.floatx(), 'int32'] + [o.dtype for o in other],
        #     parallel_iterations=self.parallel_iterations
        # )
        # l_outputs = backend.map_fn(
        #     _filter_detections,
        #     elems=[l_boxes, l_classification, other],
        #     dtype=[keras.backend.floatx(), keras.backend.floatx(), 'int32'] + [o.dtype for o in other],
        #     parallel_iterations=self.parallel_iterations
        # )

        # def _merge_panel_label(args):
        #     p_boxes = args[0]
        #     p_scores = args[1]
        #     p_labels = args[2]
        #     l_boxes = args[3]
        #     l_scores = args[4]
        #     l_labels = args[5]
        #     # return merge_panel_label_map_fn(p_boxes, p_scores, p_labels, l_boxes, l_scores, l_labels, self.merge_max_detections)
        #     return merge_panel_label_while_loop(p_boxes, p_scores, p_labels, l_boxes, l_scores, l_labels, self.merge_max_detections)

        # call filter_detections on each batch
        # outputs = backend.map_fn(
        #     _merge_panel_label,
        #     elems=p_outputs + l_outputs,
        #     dtype=[keras.backend.floatx(), keras.backend.floatx(), 'int32'] + [keras.backend.floatx(), keras.backend.floatx(), 'int32'],
        #     parallel_iterations=self.parallel_iterations
        # )

        # outputs = p_outputs + l_outputs

        return outputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [
            (input_shape[0][0], self.merge_max_detections, 4),
            (input_shape[1][0], self.merge_max_detections),
            (input_shape[1][0], self.merge_max_detections),
        ] + [
            (input_shape[0][0], self.merge_max_detections, 4),
            (input_shape[1][0], self.merge_max_detections),
            (input_shape[1][0], self.merge_max_detections),
        ]
        # return [
        #     (input_shape[0][0], self.max_detections, 4),
        #     (input_shape[1][0], self.max_detections),
        #     (input_shape[1][0], self.max_detections),
        # ] + [
        #     tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in range(2, len(input_shape))
        # ]

    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output.
        """
        # return (len(inputs) + 1) * [None]
        return (len(inputs) + 2) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms'                 : self.nms,
            'nms_threshold'       : self.nms_threshold,
            'score_threshold'     : self.score_threshold,
            'max_detections'      : self.max_detections,
            'l_nms'                 : self.l_nms,
            'l_nms_threshold'       : self.l_nms_threshold,
            'l_score_threshold'     : self.l_score_threshold,
            'l_max_detections'      : self.l_max_detections,
            'merge_max_detections'      : self.merge_max_detections,
            'parallel_iterations' : self.parallel_iterations,
        })

        return config
