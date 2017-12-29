import pprint
import random
from optparse import OptionParser
import pickle

import cv2
from keras import backend as K

import time
from keras import Input
from keras.engine import Model
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
from keras.utils import generic_utils

import Figure
from Figure import Figure
import label_rcnn_roi_helpers
import Config
import label_rcnn_data_generators
from Panel import LABEL_CLASS_MAPPING, LABEL_MAPPING
from data import get_label_rpn_data


def train_rpn(model_file=None):

    parser = OptionParser()
    parser.add_option("--train_path", dest="train_path", help="Path to training data.",
                      default='/Users/jie/projects/PanelSeg/ExpPython/train.txt')
    parser.add_option("--val_path", dest="val_path", help="Path to validation data.",
                      default='/Users/jie/projects/PanelSeg/ExpPython/eval.txt')
    parser.add_option("--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.",
                      default=32)
    parser.add_option("--network", dest="network", help="Base network to use. Supports nn_cnn_3_layer.",
                      default='nn_cnn_3_layer')
    parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.",
                      default=2000)
    parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.",
                      default='./model_rpn.hdf5')
    parser.add_option("--input_weight_path", dest="input_weight_path",
                      default='/Users/jie/projects/PanelSeg/ExpPython/models/label+bg_rpn_3_layer_color-0.135.hdf5')

    (options, args) = parser.parse_args()

    # set configuration
    c = Config.Config()

    c.model_path = options.output_weight_path
    c.num_rois = int(options.num_rois)

    import nn_cnn_3_layer as nn

    c.base_net_weights = options.input_weight_path

    val_imgs, val_classes_count = get_label_rpn_data(options.val_path)
    train_imgs, train_classes_count = get_label_rpn_data(options.train_path)

    classes_count = {k: train_classes_count.get(k, 0) + val_classes_count.get(k, 0)
                     for k in set(train_classes_count) | set(val_classes_count)}
    class_mapping = LABEL_MAPPING

    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    c.class_mapping = class_mapping

    inv_map = {v: k for k, v in class_mapping.items()}

    print('Training images per class:')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))

    config_output_filename = 'config.pickle'

    with open(config_output_filename, 'wb') as config_f:
        pickle.dump(c, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            config_output_filename))

    random.shuffle(train_imgs)
    random.shuffle(val_imgs)

    num_imgs = len(train_imgs) + len(val_imgs)

    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))

    data_gen_train = label_rcnn_data_generators.get_anchor_gt(
        train_imgs, classes_count, c, nn.nn_get_img_output_length, mode='train')
    data_gen_val = label_rcnn_data_generators.get_anchor_gt(
        val_imgs, classes_count, c, nn.nn_get_img_output_length, mode='val')

    input_shape_img = (None, None, 3)
    img_input = Input(shape=input_shape_img)
    # roi_input = Input(shape=(None, 4))

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(c.anchor_box_scales) * len(c.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)

    # classifier = nn.classifier(shared_layers, roi_input, c.num_rois, nb_classes=len(classes_count), trainable=True)

    model_rpn = Model(img_input, rpn[:2])
    # model_classifier = Model([img_input, roi_input], classifier)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    # model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    print('loading weights from {}'.format(c.base_net_weights))
    model_rpn.load_weights(c.base_net_weights, by_name=True)
    # model_classifier.load_weights(c.base_net_weights, by_name=True)
    model_rpn.summary()

    optimizer = Adam(lr=1e-5)
    # optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer, loss=[nn.rpn_loss_cls(num_anchors), nn.rpn_loss_regr(num_anchors)])
    # model_classifier.compile(optimizer=optimizer_classifier,
    #                          loss=[nn.class_loss_cls, nn.class_loss_regr(len(classes_count) - 1)],
    #                          metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    # model_all.compile(optimizer='sgd', loss='mae')

    epoch_length = 1000
    num_epochs = int(options.num_epochs)
    iter_num = 0

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()

    best_loss = np.Inf

    class_mapping_inv = {v: k for k, v in class_mapping.items()}
    print('Starting training')
    vis = True

    for epoch_num in range(num_epochs):
        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

        while True:
            try:
                if len(rpn_accuracy_rpn_monitor) == epoch_length and c.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                            mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                X, Y, img_data = next(data_gen_train)

                loss_rpn = model_rpn.train_on_batch(X, Y)

                P_rpn = model_rpn.predict_on_batch(X)

                R = label_rcnn_roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], c, K.image_dim_ordering(), use_regr=True,
                                                      overlap_thresh=0.7, max_boxes=300)
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = label_rcnn_roi_helpers.calc_iou(R, img_data, c, class_mapping)

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if c.num_rois > 1:
                    if len(pos_samples) < c.num_rois // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, c.num_rois // 2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, c.num_rois - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, c.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                # loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                #                                              [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                # losses[iter_num, 2] = loss_class[1]
                # losses[iter_num, 3] = loss_class[2]
                # losses[iter_num, 4] = loss_class[3]

                iter_num += 1

                progbar.update(iter_num,
                               [('rpn_cls', np.mean(losses[:iter_num, 0])),
                                ('rpn_regr', np.mean(losses[:iter_num, 1]))])
                # progbar.update(iter_num,
                #                [('rpn_cls', np.mean(losses[:iter_num, 0])),
                #                 ('rpn_regr', np.mean(losses[:iter_num, 1])),
                #                 ('detector_cls', np.mean(losses[:iter_num, 2])),
                #                 ('detector_regr', np.mean(losses[:iter_num, 3]))])

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    # loss_class_cls = np.mean(losses[:, 2])
                    # loss_class_regr = np.mean(losses[:, 3])
                    # class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if c.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                            mean_overlapping_bboxes))
                        # print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        # print('Loss Detector classifier: {}'.format(loss_class_cls))
                        # print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))

                    curr_loss = loss_rpn_cls + loss_rpn_regr
                    # curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if c.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                        best_loss = curr_loss
                        model_rpn.save_weights(c.model_path)
                        # model_all.save_weights(c.model_path)

                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                continue

    print('Training complete, exiting.')


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)

    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    # img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= 255

    # img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    # img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, 1.0


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


def rpn_initialize(options):
    config_output_filename = options.config_filename

    with open(config_output_filename, 'rb') as f_in:
        c = pickle.load(f_in)

    import nn_cnn_3_layer as nn

    # turn off any data augmentation at test time
    c.use_horizontal_flips = False
    c.use_vertical_flips = False
    c.rot_90 = False

    img_list_path = options.test_path

    class_mapping = c.class_mapping

    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    c.num_rois = int(options.num_rois)

    # if c.network == 'resnet50':
    #     num_features = 1024
    # elif c.network == 'vgg':
    #     num_features = 512

    input_shape_img = (None, None, 3)
    # input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    # roi_input = Input(shape=(c.num_rois, 4))
    # feature_map_input = Input(shape=input_shape_features)

    # define the base network
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(c.anchor_box_scales) * len(c.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    # classifier = nn.classifier(feature_map_input, roi_input, c.num_rois, nb_classes=len(class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    # model_classifier_only = Model([feature_map_input, roi_input], classifier)

    # model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(options.rpn_weight_path))
    model_rpn.load_weights(options.rpn_weight_path, by_name=True)
    # model_classifier.load_weights(c.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    # model_classifier.compile(optimizer='sgd', loss='mse')
    model_rpn.summary()

    model_classifier = load_model(options.classify_model_path)
    model_classifier.summary()

    all_imgs = []

    classes = {}

    bbox_threshold = 0.8

    visualise = True

    return c, model_rpn, model_classifier


def rpn_detect(figure, c, model_rpn, model_classifier):
    X, ratio = format_img(figure.image, c)

    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)

    R = label_rcnn_roi_helpers.rpn_to_roi(Y1, Y2, c, K.image_dim_ordering(), overlap_thresh=0.3, max_boxes=1000)

    R = R * c.rpn_stride

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    patches = np.empty([R.shape[0], 28, 28, 3], dtype=int)

    for idx, roi in enumerate(R):
        x, y, w, h = roi[0], roi[1], roi[2], roi[3]
        patch = figure.image[y:y + h, x:x + w]
        patches[idx] = cv2.resize(patch, (28, 28))

    patches = patches.astype('float32')
    patches[:, :, :, 0] -= c.img_channel_mean[0]
    patches[:, :, :, 1] -= c.img_channel_mean[1]
    patches[:, :, :, 2] -= c.img_channel_mean[2]
    patches /= 255

    figure.label_prediction = prediction = model_classifier.predict(patches)

    max_labels = np.argmax(prediction, 1)  # Find classified labels for each instance
    max_probs = prediction[range(len(max_labels)), max_labels]  # Find the prob of the max label
    fg_indexes = np.where(max_labels != len(LABEL_CLASS_MAPPING))  # Find foreground (label) indexes

    fg_labels = max_labels[fg_indexes]  # Find foreground labels
    fg_rois = R[fg_indexes]  # Find foreground ROIs
    fg_probs = max_probs[fg_indexes]  # Find foreground probs

    return fg_rois, fg_probs, fg_labels


def test_rpn():
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
    parser.add_option("--rpn_weight_path", dest="rpn_weight_path",
                      default='./model_rpn.hdf5') # default='/Users/jie/projects/PanelSeg/ExpPython/models/model_rpn_3_layer_color-0.0173.hdf5')
    parser.add_option("--classify_model_path", dest="classify_model_path",
                      default='/Users/jie/projects/PanelSeg/ExpPython/models/label50+bg_cnn_3_layer_color-0.9910.h5')
    parser.add_option("--result_folder", dest="result_folder",
                      default='/Users/jie/projects/PanelSeg/ExpPython/eval/rpn-1000')

    (options, args) = parser.parse_args()

    if not options.test_path:  # if filename is not given
        parser.error('Error: path to test data must be specified. Pass --path to command line')

    c, model_rpn, model_classifier = rpn_initialize(options)

    with open(options.test_path) as f:
        lines = f.readlines()

    for idx, filepath in enumerate(lines):
        print(str(idx) + ': ' + filepath)
        # if 'PMC3664797_gkt198f2p' not in filepath:
        #     continue
        # if idx < 986:
        #     continue
        filepath = filepath.strip()
        figure = Figure(filepath)
        figure.load_image()

        st = time.time()

        figure.fg_rois, figure.fg_scores, figure.fg_labels = rpn_detect(figure, c, model_rpn, model_classifier)

        # Remove overlapping candidates with the same label
        # initialize the list of picked indexes
        # idxs = np.argsort(figure.fg_probs)
        # pick = []
        # while len(idxs) > 0:
        #     # grab the last index in the indexes list and add the
        #     # index value to the list of picked indexes
        #     i = idxs[-1]
        #     label1 = figure.fg_labels[i]
        #     roi1 = figure.fg_rois[i]
        #
        #     overlap = False
        #     for idx in pick:
        #         # check whether have overlapping candidates with the same label
        #         label2 = figure.fg_labels[idx]
        #         roi2 = figure.fg_rois[idx]
        #         if label1 != label2:
        #             continue
        #         iou = label_rcnn_data_generators.iou_rect(roi1, roi2)
        #         if iou > 0:
        #             overlap = True
        #
        #     if not overlap:
        #         pick.append(i)
        #     idxs = np.delete(idxs, -1)
        # figure.fg_labels = figure.fg_labels[pick]
        # figure.fg_rois = figure.fg_rois[pick]
        # figure.fg_probs = figure.fg_probs[pick]

        print('Elapsed time = {}'.format(time.time() - st))

        # Save detection results
        figure.save_annotation(options.result_folder)


if __name__ == "__main__":
    import tensorflow as tf
    with tf.device('/cpu:0'):
        print('use CPU 0!')
        # train_rpn()
        test_rpn()

    # train_rpn()
    # test_rpn()
    pass
