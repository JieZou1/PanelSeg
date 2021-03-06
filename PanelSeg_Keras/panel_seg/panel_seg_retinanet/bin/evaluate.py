#!/usr/bin/env python

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

import argparse
import os
import sys

import keras
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import panel_seg_retinanet.bin  # noqa: F401
    __package__ = "panel_seg_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.eval import evaluate
from ..utils.keras_version import check_keras_version


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    """ Create generators for evaluation.
    """
    validation_generator = CSVGenerator(
        args.annotations,
        args.classes,
        args.l_classes,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side
    )

    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for Panel Segmentation with RetinaNet.')

    parser.add_argument('--dataset_type', help='We always use CSV only',
                        default='csv')

    parser.add_argument('--annotations', help='Path to CSV file containing annotations for evaluation.',
                        default='/Users/jie/projects/PanelSeg/programs/PanelSeg_Keras/panel_seg/panel_seg_retinanet/exp/PanelSeg/eval.csv')
    parser.add_argument('--classes', help='Path to a CSV file containing panel class mapping.',
                        default='/Users/jie/projects/PanelSeg/programs/PanelSeg_Keras/panel_seg/panel_seg_retinanet/exp/PanelSeg/mapping.csv')
    parser.add_argument('--l_classes', help='Path to a CSV file containing label class mapping.',
                        default='/Users/jie/projects/PanelSeg/programs/PanelSeg_Keras/panel_seg/panel_seg_retinanet/exp/PanelSeg/label_mapping.csv')

    parser.add_argument('--model',             help='Path to RetinaNet model.',
                        default='/Users/jie/projects/PanelSeg/programs/PanelSeg_Keras/panel_seg/panel_seg_retinanet/exp/PanelSeg/800x1333/ResNet50/not-freeze-backbone/snapshots/resnet50_csv_11.h5')

    parser.add_argument('--convert-model',   help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',        help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.3, type=float)
    parser.add_argument('--iou-threshold',   help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',  help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--l_score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.1, type=float)
    parser.add_argument('--l_iou-threshold',   help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--l_max-detections',  help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',       help='Path for saving images with detections (doesn\'t work for COCO).',
                        default='./previews')
    parser.add_argument('--image-min-side',  help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',  help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)

    return parser.parse_args(args)


def eval(generator, model, args):
    print('score_threshold: {:.4f}'.format(args.score_threshold))
    print('l_score_threshold: {:.4f}'.format(args.l_score_threshold))

    panel_aps, panel_prs, label_aps, label_prs, merge_aps, merge_prs = evaluate(
        generator,
        model,
        panel_iou_threshold=args.iou_threshold,
        panel_score_threshold=args.score_threshold,
        panel_max_detections=args.max_detections,
        label_iou_threshold=args.l_iou_threshold,
        label_score_threshold=args.l_score_threshold,
        label_max_detections=args.l_max_detections,
        save_path=args.save_path
    )

    def merge_panel_to_name(label):
        if label == generator.l_num_classes():
            return 'Panel'
        else:
            return 'Panel ' + generator.l_label_to_name(label)

    # print panel evaluation
    print_eval_results(panel_aps, panel_prs, generator.label_to_name)
    # print label evaluation
    print_eval_results(label_aps, label_prs, generator.l_label_to_name)
    # print merge evaluation
    print_eval_results(merge_aps, merge_prs, merge_panel_to_name)


def print_eval_results(ap, pr, label_to_name):
    present_classes = 0
    sum_average_precision = 0
    for label, (average_precision, num_annotations) in ap.items():
        print('{:.0f} instances of class'.format(num_annotations),
              label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
        if num_annotations > 20:
            present_classes += 1
            sum_average_precision += average_precision
    print('mAP: {:.4f}'.format(sum_average_precision / present_classes))

    total_correct = 0
    total_detections = 0
    total_annotations = 0
    for label, (correct, num_detections, num_annotations) in pr.items():
        if num_annotations == 0:
            precision, recall = 0, 0
        elif num_detections == 0:
            precision, recall = 0, 0
        else:
            precision, recall = correct / num_detections, correct / num_annotations
        print('{:.0f} instances of class'.format(num_annotations),
              label_to_name(label), 'with precision: {:.4f}, recall: {:.4f}'.format(precision, recall))
        total_correct += correct
        total_detections += num_detections
        total_annotations += num_annotations
    print('Precision: {:.4f}'.format(total_correct / total_detections))
    print('Recall: {:.4f}'.format(total_correct / total_annotations))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    generator = create_generator(args)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone, convert=args.convert_model)

    # print model summary
    # print(model.summary())

    # start evaluation
    if args.dataset_type == 'coco':
        from ..utils.coco_eval import evaluate_coco
        evaluate_coco(generator, model, args.score_threshold)
    else:
        eval(generator, model, args)


if __name__ == '__main__':
    main()
