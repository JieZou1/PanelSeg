from optparse import OptionParser

import time
import os
import numpy as np

import Panel
from Figure import Figure
from label_hog import hog_initialize, hog_detect
from label_rcnn_data_generators import iou_rect
from label_rpn import rpn_initialize, rpn_detect

HOG_ONLY = 0
RPN_ONLY = 1
HOG_RPN = 2


def rpn_hog_initialize(options):
    rpn_c, rpn_model_rpn, rpn_model_classifier = rpn_initialize(options)
    hog = hog_initialize()

    return rpn_c, rpn_model_rpn, rpn_model_classifier, hog


def rpn_hog_detect(figure, rpn_c, rpn_model_rpn, rpn_model_classifier, hog, method):

    if method == HOG_ONLY:
        hog_rois, hog_scores, hog_labels = hog_detect(figure, hog)
        rois, scores, labels = hog_rois, hog_scores, hog_labels
    elif method == RPN_ONLY:
        rpn_rois, rpn_scores, rpn_labels = rpn_detect(figure, rpn_c, rpn_model_rpn, rpn_model_classifier)
        rois, scores, labels = rpn_rois, rpn_scores, rpn_labels
    elif method == HOG_RPN:
        hog_rois, hog_scores, hog_labels = hog_detect(figure, hog)
        rpn_rois, rpn_scores, rpn_labels = rpn_detect(figure, rpn_c, rpn_model_rpn, rpn_model_classifier)

        # Keep all regions detected by both RPN and HOG methods.
        rois, scores, labels = rpn_hog_combine_all(hog_rois, hog_scores, hog_labels, rpn_rois, rpn_scores, rpn_labels)

        # Keep only regions agreed by both methods. IOU > 25%
        # rois, scores, labels = rpn_hog_combine_agreed(hog_rois, hog_scores, hog_labels,
        #                                               rpn_rois, rpn_scores, rpn_labels, 0.25)

    return rois, scores, labels


def rpn_hog_combine_all(hog_rois, hog_scores, hog_labels, rpn_rois, rpn_scores, rpn_labels):
    """
    Keep all regions detected by both RPN and HOG methods.
    This shows the upper limit of recalls
    :return: combined results
    """
    if hog_rois is None and rpn_rois is None:
        rois, scores, labels = None, None, None
    elif hog_rois is None:
        rois, scores, labels = rpn_rois, rpn_scores, rpn_labels
    elif rpn_rois is None:
        rois, scores, labels = hog_rois, hog_scores, hog_labels
    else:
        rois = np.concatenate((hog_rois, rpn_rois), axis=0)
        scores = np.concatenate((hog_scores.reshape(hog_scores.shape[0],), rpn_scores), axis=0)
        labels = np.concatenate((hog_labels, rpn_labels), axis=0)

    return rois, scores, labels


def rpn_hog_combine_agreed(hog_rois, hog_scores, hog_labels, rpn_rois, rpn_scores, rpn_labels, iou_threshold):
    """
    Keep only regions agreed by both methods. IOU > iou_threshold
    :return: combined results
    """
    if hog_rois is None and rpn_rois is None:
        rois, scores, labels = None, None, None
    elif hog_rois is None:
        rois, scores, labels = rpn_rois, rpn_scores, rpn_labels
    elif rpn_rois is None:
        rois, scores, labels = hog_rois, hog_scores, hog_labels
    else:
        indexes = []
        for hog_i in range(len(hog_labels)):
            hog_roi = hog_rois[hog_i]
            hog_label = hog_labels[hog_i]
            hog_score = hog_scores[hog_i]

            for rpn_i in range(len(rpn_labels)):
                rpn_roi = rpn_rois[rpn_i]
                rpn_label = rpn_labels[rpn_i]
                rpn_score = rpn_scores[rpn_i]

                iou = iou_rect(rpn_roi, hog_roi)
                if iou < iou_threshold:
                    continue

                indexes.append(hog_i)
                break

        rois = hog_rois[indexes]
        labels = hog_labels[indexes]
        scores = hog_scores[indexes]

    return rois, scores, labels


def test_rpn_hog():
    """
    HOG+RPN for panel label recognition
    :return:
    """
    parser = OptionParser()

    parser.add_option("-p", "--path", dest="test_path", help="Path to test data.",
                      default='/Users/jie/projects/PanelSeg/ExpPython/eval.txt')
    parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
                      help="Number of ROIs per iteration. Higher means more memory use.", default=32)
    parser.add_option("--config_filename", dest="config_filename",
                      help="Location to read the metadata related to the training (generated when training).",
                      default="config.pickle")
    parser.add_option("--network", dest="network", help="Base network to use. Supports nn_cnn_3_layer.",
                      default='nn_cnn_3_layer')
    parser.add_option("--rpn_weight_path", dest="rpn_weight_path",  default='./model_rpn.hdf5')
                      # default='/Users/jie/projects/PanelSeg/ExpPython/models/label+bg_rpn_3_layer_color-0.0374.hdf5')
    parser.add_option("--classify_model_path", dest="classify_model_path",
                      default='/Users/jie/projects/PanelSeg/ExpPython/models/label50+bg_cnn_3_layer_color-0.9910.h5')
    parser.add_option("--result_folder", dest="result_folder",
                      default='/Users/jie/projects/PanelSeg/ExpPython/eval/rpn_hog/rpn_cpu_0.116')

    (options, args) = parser.parse_args()

    if not options.test_path:  # if filename is not given
        parser.error('Error: path to test data must be specified. Pass --path to command line')

    rpn_c, rpn_model_rpn, rpn_model_classifier, hog = rpn_hog_initialize(options)

    with open(options.test_path) as f:
        lines = f.readlines()

    for idx, filepath in enumerate(lines):
        print(str(idx) + ': ' + filepath)
        # if 'PMC3664797_gkt198f2p' not in filepath:
        #     continue
        # if idx < 243:
        #     continue
        filepath = filepath.strip()
        figure = Figure(filepath)
        figure.load_image()

        st = time.time()

        figure.fg_rois, figure.fg_scores, figure.fg_labels = rpn_hog_detect(figure,
                                                                            rpn_c, rpn_model_rpn, rpn_model_classifier,
                                                                            hog, RPN_ONLY)

        print('Elapsed time = {}'.format(time.time() - st))

        # Save detection results
        figure.save_annotation(options.result_folder)


def combine_hog_rpn():
    """
    Load results from HOG and RPN methods and then combine them
    :return:
    """
    parser = OptionParser()

    parser.add_option("-p", "--path", dest="test_path", help="Path to test data.",
                      default='/Users/jie/projects/PanelSeg/ExpPython/eval.txt')
    parser.add_option("--hog_folder", dest="hog_folder",
                      default='/Users/jie/projects/PanelSeg/ExpPython/eval/rpn_hog/hog')
    parser.add_option("--rpn_folder", dest="rpn_folder",
                      default='/Users/jie/projects/PanelSeg/ExpPython/eval/rpn_hog/rpn_cpu_0.0374')
    parser.add_option("--result_folder", dest="result_folder",
                      default='/Users/jie/projects/PanelSeg/ExpPython/eval/rpn_hog/hog_rpn')

    (options, args) = parser.parse_args()

    with open(options.test_path) as f:
        lines = f.readlines()

    for idx, filepath in enumerate(lines):
        print(str(idx) + ': ' + filepath)
        # if 'PMC3664797_gkt198f2p' not in filepath:
        #     continue
        # if idx < 243:
        #     continue
        filepath = filepath.strip()
        figure = Figure(filepath)
        figure.load_image()

        # Load HOG result
        hog_file_path = os.path.join(options.hog_folder, figure.file).replace('.jpg', '_data.xml')
        figure.load_annotation(hog_file_path)
        hog_rois = np.empty([len(figure.panels), 4], dtype=int)
        for i in range(len(figure.panels)):
            hog_rois[i] = figure.panels[i].label_rect
        hog_rois[:, 0] += Figure.PADDING
        hog_rois[:, 1] += Figure.PADDING
        hog_labels = np.full(hog_rois.shape[0], Panel.LABEL_ALL)
        hog_scores = np.full(hog_rois.shape[0], 1.0)

        # Load RPN result
        rpn_file_path = os.path.join(options.rpn_folder, figure.file).replace('.jpg', '_data.xml')
        figure.load_annotation(rpn_file_path)
        rpn_rois = np.empty([len(figure.panels), 4], dtype=int)
        for i in range(len(figure.panels)):
            rpn_rois[i] = figure.panels[i].label_rect
        rpn_rois[:, 0] += Figure.PADDING
        rpn_rois[:, 1] += Figure.PADDING
        rpn_labels = np.full(rpn_rois.shape[0], Panel.LABEL_ALL)
        rpn_scores = np.full(rpn_rois.shape[0], 1.0)

        # Keep all regions detected by both RPN and HOG methods.
        # rois, scores, labels = rpn_hog_combine_all(hog_rois, hog_scores, hog_labels, rpn_rois, rpn_scores, rpn_labels)

        # Keep only regions agreed by both methods. IOU > 25%
        rois, scores, labels = rpn_hog_combine_agreed(hog_rois, hog_scores, hog_labels,
                                                      rpn_rois, rpn_scores, rpn_labels, 0.125)

        figure.fg_rois, figure.fg_scores, figure.fg_labels = rois, scores, labels

        # Save detection results
        figure.save_annotation(options.result_folder)


if __name__ == "__main__":
    import tensorflow as tf
    with tf.device('/cpu:0'):
        print('use CPU 0!')
        test_rpn_hog()
        # combine_hog_rpn()
