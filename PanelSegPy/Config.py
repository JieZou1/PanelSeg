import os

import cv2


class Config:
    def __init__(self):

        self.root_folder = '/Users/jie/projects/PanelSeg/ExpRcnn'
        self.list_file = os.path.join(self.root_folder, 'all.txt')
        self.labels_folder = os.path.join(self.root_folder, 'Labels')
        self.labels_normalized_folder = os.path.join(self.root_folder, 'Labels-28')
        self.nonlabels_folder = os.path.join(self.root_folder, 'NonLabels')
        self.nonlabels_normalized_folder = os.path.join(self.root_folder, 'NonLabels-28')
        self.color_type = cv2.IMREAD_COLOR

        # Below are RCNN configurations



        self.verbose = True

        # setting for data augmentation
        # self.use_horizontal_flips = False
        # self.use_vertical_flips = False
        # self.rot_90 = False

        # anchor box scales
        # self.anchor_box_scales = [128, 256, 512]
        self.anchor_box_scales = [8, 12, 16, 24, 32, 40, 48, 56, 64]

        # anchor box ratios
        # self.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]
        self.anchor_box_ratios = [[1, 1]]

        # size to resize the smallest side of the image
        self.im_size = 600

        # image channel-wise mean to subtract
        # self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_channel_mean = [128.0, 128.0, 128.0]
        self.img_scaling_factor = 1.0
        self.img_pixel_mean = 128.0

        # number of ROIs at once
        self.num_rois = 4

        # stride at the RPN (this depends on the network configuration)
        self.rpn_stride = 4

        self.balanced_classes = False

        # scaling the stdev
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.6

        # overlaps for classifier ROIs
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        # placeholder for the class mapping, automatically generated by the parser
        self.class_mapping = None

        #location of pretrained weights for the base network
        # weight files can be found at:
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

        self.model_path = 'model_frcnn.vgg.hdf5'
