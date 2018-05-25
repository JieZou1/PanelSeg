import csv
import sys
import argparse
import time
import os

import numpy as np
import tensorflow as tf
import cv2
from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box

from figure import misc
from figure.figure import Figure
from figure.figure_set import FigureSet


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def parse_args(args):
    parser = argparse.ArgumentParser(description='RetinaNet for panel splitting.')
    parser.add_argument('--model', help='Path to the trained model file.',
                        default='/Users/jie/projects/PanelSeg/ExpKeras/panel_split/rentinanet/train-with-clef2016/snapshots/resnet50_csv_50.h5'
                        # default='/Users/jie/projects/PanelSeg/ExpKeras/panel_split/rentinanet/train-with-ours/snapshots/resnet50_csv_40.h5'
                        )
    parser.add_argument('--eval_list', help='Path to the evaluation list file.',
                        default='/Users/jie/projects/PanelSeg/ExpKeras/eval.txt'
                        # default='/Users/jie/projects/PanelSeg/ExpKeras/clef_eval.txt'
                        )

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Load figure set
    print('Evaluation list file is {0}'.format(args.eval_list))
    figure_files = misc.read_sample_list(args.eval_list)

    # Load model
    print('Load model from {0}'.format(args.model))
    model = models.load_model(args.model, convert=True)
    model.summary()

    mean_time = 0
    for figure_idx, figure_path in enumerate(figure_files):
        if not os.path.exists(figure_path):
            print('{0} does not exist!'.format(figure_path))
            continue

        image = read_image_bgr(figure_path)
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        start = time.time()
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale

        processing_time = time.time() - start
        print('processing {0}:{1} time: {2}'.format(figure_idx, figure_path, processing_time))
        mean_time += processing_time

        #save results
        figure_folder, figure_file = os.path.split(figure_path)
        image_path = os.path.join('results', figure_file)
        csv_path = image_path.replace('.jpg', '.csv')

        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')

            for idx, (box, score, label) in enumerate(zip(boxes[0], scores[0], labels[0])):
                # scores are sorted so we can break
                if score < 0.5:
                    break

                color = label_color(idx)
                b = box.astype(int)
                draw_box(draw, b, color)

                csv_writer.writerow([figure_path,
                                    str(b[0]), str(b[1]), str(b[2]), str(b[3]),
                                    'panel'])

        cv2.imwrite(image_path, draw)

    print('average processing time: {0}'.format(mean_time / len(figure_files)))


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()
    # main()
