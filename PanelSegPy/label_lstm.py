from optparse import OptionParser

import cv2
import time
from keras import Sequential
from keras import layers
import numpy as np
import os
import pickle
import svmutil

from keras.models import load_model
from keras.utils import to_categorical

from Figure import Figure
from Panel import LABEL_CLASS_MAPPING, Panel, map_label, CLASS_LABEL_MAPPING, case_same_label
from label_classify_50 import cnn_feature_extraction
from label_hog import hog_feature_extraction
from label_rcnn_data_generators import iou_rect

RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1


def create_lstm_model(num_features):
    model = Sequential()
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
    # Note: In a situation where your input sequences have a variable length,
    # use input_shape=(None, num_feature).
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE, input_shape=(None, num_features), return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.
    model.add(layers.TimeDistributed(layers.Dense(len(LABEL_CLASS_MAPPING) + 1)))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def load_ground_truth_annotation(figure):
    figure.load_gt_annotation(which_annotation='label')
    gt_rois = np.empty([len(figure.panels), 4], dtype=int)
    gt_labels = np.full(gt_rois.shape[0], '')
    for j in range(len(figure.panels)):
        gt_rois[j] = figure.panels[j].label_rect
        gt_labels[j] = figure.panels[j].label
    gt_rois[:, 0] += Figure.PADDING
    gt_rois[:, 1] += Figure.PADDING
    return gt_rois, gt_labels


def load_auto_annotation(figure, auto_folder):
    auto_file_path = os.path.join(auto_folder, figure.file).replace('.jpg', '_data.xml')
    figure.load_annotation(auto_file_path)
    auto_rois = np.empty([len(figure.panels), 4], dtype=int)
    for j in range(len(figure.panels)):
        auto_rois[j] = figure.panels[j].label_rect
    auto_rois[:, 0] += Figure.PADDING
    auto_rois[:, 1] += Figure.PADDING
    return auto_rois


def feature_extraction(figure, rois):

    F = []

    patches = np.empty([rois.shape[0], 64, 64], dtype=np.uint8)
    for idx, roi in enumerate(rois):
        x, y, w, h = roi[0], roi[1], roi[2], roi[3]

        # extract hog features
        patch = figure.image_gray[y:y + h, x:x + w]
        patches[idx] = cv2.resize(patch, (64, 64))
        hog_f = hog_feature_extraction(patches[idx])
        # cnn_f = cnn_feature_extraction(figure, roi)

        # extract position and size features
        pos_f = np.array([x / figure.image_width, y / figure.image_height, w / 100.0])

        # f = np.append(hog_f, pos_f)
        # F.append(f)
        F.append(hog_f)
    return np.array(F)


def read_samples(path, auto_folder, model_classifier, model_svm):

    with open(path) as f:
        lines = f.readlines()
    # Remove whitespace characters, and then construct the figures
    figures = [Figure(line.strip()) for line in lines]

    X = []
    Y = []
    for i, figure in enumerate(figures):
        figure.load_image()

        # load ground-truth annotation
        gt_rois, gt_labels = load_ground_truth_annotation(figure)

        # load auto annotation
        auto_rois = load_auto_annotation(figure, auto_folder)

        # sort auto annotation with respect to distances to left-up corner (0, 0)
        distances = [roi[0] + roi[1] for roi in auto_rois]
        indexes = np.argsort(distances)
        auto_rois = auto_rois[indexes]

        # match auto to gt to assign y
        y = np.full([auto_rois.shape[0]], len(LABEL_CLASS_MAPPING))  # initialize as non-label
        for gt_i, gt_roi in enumerate(gt_rois):
            ious = [iou_rect(auto_roi, gt_roi) for auto_roi in auto_rois]
            max_index = np.argmax(ious)
            if ious[max_index] > 0.25:
                y[max_index] = LABEL_CLASS_MAPPING[map_label(gt_labels[gt_i])]
        Y.append(y)

        # extract features
        x = feature_extraction(figure, auto_rois)

        # if len(x) > 0:
        #     p_label, p_acc, p_val = svmutil.svm_predict(y, x.tolist(), model_svm, '-b 1')
        #     x = np.array(p_val)

        X.append(x)

    return X, Y


def train_lstm():

    print('lstm training:')

    parser = OptionParser()

    parser.add_option("--train_path", dest="train_path", help="Path to training data.",
                      default='/Users/jie/projects/PanelSeg/ExpPython/train.txt')
    parser.add_option("--train_auto_folder", dest="train_auto_folder",
                      default='/Users/jie/projects/PanelSeg/ExpPython/eval/rpn_hog/train_rpn')
    parser.add_option("--eval_path", dest="eval_path", help="Path to eval data.",
                      default='/Users/jie/projects/PanelSeg/ExpPython/eval.txt')
    parser.add_option("--eval_auto_folder", dest="eval_auto_folder",
                      default='/Users/jie/projects/PanelSeg/ExpPython/eval/rpn_hog/rpn')
    # parser.add_option("--classify_model_path", dest="classify_model_path",
    #                   default='/Users/jie/projects/PanelSeg/ExpPython/models/label50+bg_cnn_3_layer_color-0.9910.h5')
    # parser.add_option("--svm_model_path", dest="svm_model_path",
    #                   default='/Users/jie/projects/PanelSeg/Exp/LabelClassifySvmTrain/SVMModel-51classes-with-neg/svm_model_rbf_8.0_0.125')

    (options, args) = parser.parse_args()

    # model_classifier = load_model(options.classify_model_path)
    # model_classifier.summary()
    model_classifier = None

    # model_svm = svmutil.svm_load_model(options.svm_model_path)
    model_svm = None

    X_eval, Y_eval = read_samples(options.eval_path, options.eval_auto_folder, model_classifier, model_svm)
    # with open('Eval.pickle', 'wb') as file:
    #     pickle.dump((X_eval, Y_eval), file)
    # with open('Eval.pickle', 'rb') as file:
    #     (X_eval, Y_eval) pickle.load(file)

    X_train, Y_train = read_samples(options.train_path, options.train_auto_folder, model_classifier, model_svm)
    # with open('Train.pickle', 'wb') as file:
    #     pickle.dump((X_train, Y_train), file)
    # with open('Train.pickle', 'rb') as file:
    #     (X_train, Y_train) pickle.load(file)

    # X_train += X_eval[:int(len(X_eval)/4)]
    # Y_train += Y_eval[:int(len(Y_eval)/4)]

    model = create_lstm_model(X_eval[0].shape[1])

    # Because of variable length of the sequence, we train with batch_size = 1
    n_epoch = 10
    for i in range(n_epoch):
        print('epoch {0}:'.format(i))
        for j, (x, y) in enumerate(zip(X_train, Y_train)):
            if y.size == 0:
                continue
            print('Epoch {0}, sample {1}:'.format(i, j))
            _x = np.expand_dims(x, axis=0)
            _y = np.expand_dims(to_categorical(y, num_classes=len(LABEL_CLASS_MAPPING)+1), axis=0)
            model.fit(_x, _y, epochs=1, batch_size=1, verbose=1)

    # evaluate on eval data
    # total = 0
    # correct = 0
    # for x, y in zip(X_eval, Y_eval):
    #     if y.size == 0:
    #         continue
    #     _x = np.expand_dims(x, axis=0)
    #     y_hat = model.predict(_x)
    #     y_max = [np.argmax(y_t) for y_t in y_hat[0]]

    model.save('lstm_model_epoch_{0}_train_0.25.h5'.format(i))


def load_samples(path):
    with open(path) as f:
        lines = f.readlines()
    # Remove whitespace characters, and then construct the figures
    figures = [Figure(line.strip()) for line in lines]
    return figures


def max_y_hat(rois, y_hats):

    max_rois = []
    max_probs = []
    max_labels = []
    for i in range(rois.shape[0]):
        y_hat = y_hats[i]
        y_max = np.argmax(y_hat)
        if y_max == len(LABEL_CLASS_MAPPING):
            continue
        max_rois.append(rois[i])
        max_probs.append(y_hat[y_max])
        max_labels.append(y_max)

    return max_rois, max_probs, max_labels


class BeamItem:
    def __init__(self):
        self.log_prob = float("inf")
        self.score = float("inf")
        self.label_indexes = []
        self.rois = []


def duplicate_labels(beam_item):
    labels = []
    for label_index in beam_item.label_indexes:
        if label_index == len(LABEL_CLASS_MAPPING):
            continue
        label = CLASS_LABEL_MAPPING[label_index]
        label_lower = label.lower()
        if label_lower in labels:
            return True
        labels.append(label_lower)
    return False


def overlapping_rect(beam_item):
    last_label_index = beam_item.label_indexes[-1]
    prev_label_indexes = beam_item.label_indexes[:-1]

    if last_label_index == len(LABEL_CLASS_MAPPING):
        return False

    prev_rects = beam_item.rois[:-1]
    last_rect = beam_item.rois[-1]
    for i, rect in enumerate(prev_rects):
        if prev_label_indexes[i] == len(LABEL_CLASS_MAPPING):
            continue
        iou = iou_rect(rect, last_rect)
        if iou > 0:
            return True
    return False


def same_case_labels(beam_item):
    labels = []
    for label_index in beam_item.label_indexes:
        if label_index == len(LABEL_CLASS_MAPPING):
            continue
        label = CLASS_LABEL_MAPPING[label_index]
        labels.append(label)

    sequence_type = label_type = ''
    for label in labels:
        if case_same_label(label):
            continue

        if len(sequence_type) == 0:
            if label.isdigit():
                sequence_type = 'digit'
            elif label.islower():
                sequence_type = 'lower'
            elif label.isupper():
                sequence_type = 'upper'
        else:
            if label.isdigit():
                label_type = 'digit'
            elif label.islower():
                label_type = 'lower'
            elif label.isupper():
                label_type = 'upper'
            if sequence_type != label_type:
                return False
    return True


def is_valid_label_sequence(beam_item):
    if overlapping_rect(beam_item):
        return False
    if duplicate_labels(beam_item):
        return False
    if not same_case_labels(beam_item):
        return False
    return True


def update_score(beam_item):
    beam_item.score = beam_item.log_prob / len(beam_item.label_indexes)


def cut_beam(beam, beam_length):
    """
    Sort beam and pick the top beam_length item
    :param beam:
    :return:
    """

    scores = []
    for item in beam:
        scores.append(item.score)
    sorted_indexes = np.argsort(scores)[::-1]

    sorted_beam = np.array(beam)[sorted_indexes]

    if len(beam) > beam_length:
        return sorted_beam[:beam_length]
    else:
        return sorted_beam


def beam_search_with_neg(rois, y_hats, beam_length):

    beams = []
    for i in range(rois.shape[0]):
        y_hat = y_hats[i]
        beam = []
        if i == 0:
            for j in range(len(y_hat)):
                if j < len(CLASS_LABEL_MAPPING) and y_hat[j] < 0.0001:
                    continue
                beam_item = BeamItem()
                beam_item.log_prob = np.log(y_hat[j])
                beam_item.label_indexes.append(j)
                beam_item.rois.append(rois[i])

                if not is_valid_label_sequence(beam_item):
                    continue

                update_score(beam_item)
                beam.append(beam_item)
        else:
            prev_beam = beams[-1]
            for k in range(len(prev_beam)):
                prev_item = prev_beam[k]
                for j in range(len(y_hat)):
                    if j < len(CLASS_LABEL_MAPPING) and y_hat[j] < 0.0001:
                        continue
                    beam_item = BeamItem()
                    beam_item.log_prob = prev_item.log_prob + np.log(y_hat[j])
                    beam_item.label_indexes.extend(prev_item.label_indexes)
                    beam_item.label_indexes.append(j)
                    beam_item.rois.extend(prev_item.rois)
                    beam_item.rois.append(rois[i])

                    if not is_valid_label_sequence(beam_item):
                        continue

                    update_score(beam_item)
                    beam.append(beam_item)

        beam = cut_beam(beam, beam_length)
        beams.append(beam)

    last_beam = beams[-1]

    max_beam_item = last_beam[0]

    max_rois = []
    max_probs = []
    max_labels = []
    for i, label_index in enumerate(max_beam_item.label_indexes):
        if label_index == len(LABEL_CLASS_MAPPING):
            continue
        max_rois.append(rois[i])
        max_probs.append(y_hats[i][label_index])
        max_labels.append(label_index)

    return max_rois, max_probs, max_labels


def beam_search_without_neg(rois, y_hats, beam_length):
    # TODO: implement beam search without neg samples (collect only positive labels in the beams)
    beams = []
    for i in range(rois.shape[0]):
        y_hat = y_hats[i]

        beam = []
        for j in range(len(y_hat) - 1):  # we do not care the prob of Non-Label (neg)
            prob = y_hat[j]
            if prob < 0.01:
                continue    # We do not care if post-prob is too small

            if i == 0:
                beam_item = BeamItem()
                beam_item.label_indexes.append(j)
                beam_item.score = np.log(prob)
                if not is_valid_label_sequence(beam_item):
                    continue
                update_score(beam_item)
                beam.append(beam_item)
            else:
                pass

    pass


def test_lstm():
    parser = OptionParser()

    parser.add_option("--eval_path", dest="eval_path", help="Path to eval data.",
                      default='/Users/jie/projects/PanelSeg/ExpPython/eval.txt')
    parser.add_option("--eval_auto_folder", dest="eval_auto_folder",
                      default='/Users/jie/projects/PanelSeg/ExpPython/eval/rpn_hog/rpn')
    parser.add_option("--classify_model_path", dest="classify_model_path",
                      default='/Users/jie/projects/PanelSeg/ExpPython/models/label50+bg_cnn_3_layer_color-0.9910.h5')
    # parser.add_option("--svm_model_path", dest="svm_model_path",
    #                   default='/Users/jie/projects/PanelSeg/Exp/LabelClassifySvmTrain/SVMModel-51classes-with-neg/svm_model_rbf_8.0_0.125')
    parser.add_option("--lstm_model_path", dest="lstm_model_path",
                      default='/Users/jie/projects/PanelSeg/ExpPython/models/lstm_model_train_0.25eval_epoch_9.h5')
    parser.add_option("--result_folder", dest="result_folder",
                      default='/Users/jie/projects/PanelSeg/ExpPython/eval/rpn_hog_lstm/eval')

    (options, args) = parser.parse_args()

    # model_classifier = load_model(options.classify_model_path)
    # model_classifier.summary()

    # model_svm = svmutil.svm_load_model(options.svm_model_path)
    model_svm = None

    model_lstm = load_model(options.lstm_model_path)
    model_lstm.summary()

    with open(options.eval_path) as f:
        lines = f.readlines()

    for idx, filepath in enumerate(lines):
        print(str(idx) + ': ' + filepath)
        # if '1757-1626-0002-0000008402-001' not in filepath:
        #     continue
        # if idx < 37:
        #     continue
        filepath = filepath.strip()
        figure = Figure(filepath)
        figure.load_image()
        st = time.time()

        # load detection results by RPN
        rois = load_auto_annotation(figure, options.eval_auto_folder)

        # sort auto annotation with respect to distances to left-up corner (0, 0)
        distances = [roi[0] + roi[1] for roi in rois]
        indexes = np.argsort(distances)
        rois = rois[indexes]

        x = feature_extraction(figure, rois)

        if rois.size == 0:
            figure.fg_rois, figure.fg_scores, figure.fg_labels = None, None, None
        else:
            # x = x.tolist()
            # y = np.zeros(len(x)).tolist()
            # p_label, p_acc, p_val = svmutil.svm_predict(y, x, model_svm, '-b 1')
            # x = np.array(p_val)

            _x = np.expand_dims(x, axis=0)
            y_hat = model_lstm.predict(_x)

            # figure.fg_rois, figure.fg_scores, figure.fg_labels = max_y_hat(rois, y_hat[0])
            figure.fg_rois, figure.fg_scores, figure.fg_labels = beam_search_with_neg(rois, y_hat[0], 5)

        print('Elapsed time = {}'.format(time.time() - st))

        # Save detection results
        figure.save_annotation(options.result_folder)


if __name__ == "__main__":
    import tensorflow as tf
    with tf.device('/gpu:0'):
        print('use GPU 0!')
        # train_lstm()
        test_lstm()
