import csv
import sys
import argparse
import time
import os

import keras
import numpy as np
import tensorflow as tf
import cv2
import xml.etree.cElementTree as ET

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import panel_split_retinanet.bin
    __package__ = "panel_split_retinanet.bin"

from .. import models
from ..utils.colors import label_color
from ..utils.image import read_image_bgr, preprocess_image, resize_image
from ..utils.visualization import draw_box


def get_session():
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    return tf.Session(config=config)


def parse_args(args):
    parser = argparse.ArgumentParser(description='RetinaNet for panel splitting.')
    parser.add_argument('--backbone', help='The Backbone Model',
                        default='resnet152'
                        )
    parser.add_argument('--model', help='Path to the trained model file.',
                        # default='/Users/jie/projects/PanelSeg/programs/PanelSeg_Keras/snapshots/vgg16_csv_03.h5'
                        default='/Users/jie/projects/PanelSeg/ExpKeras/panel_split/retinanet/clef2016/ResNet50/snapshots/resnet50_csv_08.h5'
                        # default='/Users/jie/projects/PanelSeg/ExpKeras/panel_split/rentinanet/train-with-ours/snapshots/resnet50_csv_40.h5'
                        )
    parser.add_argument('--eval_list', help='Path to the evaluation list file.',
                        # default='/Users/jie/projects/PanelSeg/ExpKeras/eval.txt'
                        default='/Users/jie/projects/PanelSeg/ExpKeras/clef_eval.txt'
                        )

    return parser.parse_args(args)


def overlap_with_mask(box, mask):
    roi = mask[box[1]:box[3], box[0]:box[2]]
    count = np.count_nonzero(roi)
    box_area = (box[2] - box[0])*(box[3]-box[1])
    ratio = count/box_area
    return ratio


def update_mask(box, mask):
    mask[box[1]:box[3], box[0]:box[2]] = 1


def predict(model, image):
    # Call original model to predict
    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale

    boxes, scores, labels = boxes[0], scores[0], labels[0]

    idxs = list()
    for idx, score in enumerate(scores):
        if score > 0.5:
            idxs.append(idx)
        else:
            break   # scores are sorted so we can break
    boxes, scores, labels = boxes[idxs], scores[idxs], labels[idxs]

    # # Post processing to remove false positives
    # # 1. We keep only scores greater than 0.05
    # idxs = list()
    # for idx, score in enumerate(scores):
    #     if score > 0.05:
    #         idxs.append(idx)
    #     else:
    #         break   # scores are sorted so we can break
    # boxes, scores, labels = boxes[idxs], scores[idxs], labels[idxs]
    #
    # # 2. We remove boxes which overlaps with other high score boxes
    # idxs = list()
    # height, width, depth = image.shape
    # mask = np.zeros((height, width))
    # for idx, box in enumerate(boxes):
    #     box_i = box.astype(int)
    #     if overlap_with_mask(box_i, mask) < 0.66:
    #         idxs.append(idx)
    #         update_mask(box_i, mask)
    #
    # boxes, scores, labels = boxes[idxs], scores[idxs], labels[idxs]
    #
    # # 3. We keep only at  most 30 panels
    # if len(boxes) > 30:
    #     boxes, scores, labels = boxes[:30], scores[:30], labels[:30]

    return boxes, scores, labels


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    keras.backend.tensorflow_backend.set_session(get_session())

    # Load figure set
    print('Evaluation list file is {0}'.format(args.eval_list))
    # figure_files = misc.read_sample_list(args.eval_list)
    with tf.gfile.GFile(args.eval_list) as fid:
        lines = fid.readlines()
    figure_files = [line.strip().split(' ')[0] for line in lines]

    # Load model
    print('Load model from {0}'.format(args.model))
    model = models.load_model(args.model, backbone_name=args.backbone, convert=True)
    model.summary()

    # prepare ImageCLEF XML file
    annotations_node = ET.Element("annotations")

    tree = ET.ElementTree(annotations_node)

    mean_time = 0
    for figure_idx, figure_path in enumerate(figure_files):
        if not os.path.exists(figure_path):
            print('{0} does not exist!'.format(figure_path))
            continue

        # if '1471-213X-8-30-5' not in figure_path:
        #     continue

        image = read_image_bgr(figure_path)
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes, scores, labels = predict(model, image)
        processing_time = time.time() - start
        print('processing {0}:{1} time: {2}'.format(figure_idx, figure_path, processing_time))
        mean_time += processing_time

        # save results
        figure_folder, figure_file = os.path.split(figure_path)
        image_path = os.path.join('results', figure_file)
        csv_path = image_path.replace('.jpg', '.csv')

        # save to ImageCLEF XML file
        annotation_node = ET.SubElement(annotations_node, "annotation")
        filename_node = ET.SubElement(annotation_node, "filename").text = os.path.splitext(figure_file)[0]

        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')

            for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                # draw preview image
                color = label_color(idx)
                b = box.astype(int)
                draw_box(draw, b, color)

                # save to individual csv file
                csv_writer.writerow([figure_path,
                                    str(b[0]), str(b[1]), str(b[2]), str(b[3]),
                                    'panel'])

                object_node = ET.SubElement(annotation_node, "object")
                point_tl_node = ET.SubElement(object_node, "point", x=str(b[0]), y=str(b[1]))
                point_tr_node = ET.SubElement(object_node, "point", x=str(b[2]), y=str(b[1]))
                point_bl_node = ET.SubElement(object_node, "point", x=str(b[0]), y=str(b[3]))
                point_br_node = ET.SubElement(object_node, "point", x=str(b[2]), y=str(b[3]))

        cv2.imwrite(image_path, draw)

    tree.write("results.xml")
    print('average processing time: {0}'.format(mean_time / len(figure_files)))


if __name__ == '__main__':
    # with tf.device('/cpu:0'):
    #     main()
    main()
