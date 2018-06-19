import csv
import cv2
import sys
import os
import argparse

import xml.etree.ElementTree as ET

from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.visualization import draw_box


def parse_args(args):
    parser = argparse.ArgumentParser(description='Convert ImageCLEF PanelSeg data set to CSV format.')
    parser.add_argument('--data_root',
                        help='The original ImageCLEF GT.XML file.',
                        # default='/datasets/ImageCLEF/2016/test',
                        default='/datasets/ImageCLEF/2016/training',
                        type=str)
    parser.add_argument('--gt_path',
                        help='The original ImageCLEF GT.XML file.',
                        # default='/datasets/ImageCLEF/2016/test/FigureSeparationTest2016GT.xml',
                        default='/datasets/ImageCLEF/2016/training/FigureSeparationTraining2016-GT.xml',
                        type=str)
    parser.add_argument('--csv_path',
                        help='The converted csv file path.',
                        default='train.csv',
                        type=str)
    parser.add_argument('--list_path',
                        help='The figure image file list.',
                        default='train.txt',
                        type=str)
    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    image_paths = list()

    with open(args.csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')



        tree = ET.parse(args.gt_path)
        # get root element
        root = tree.getroot()
        annotation_items = root.findall('./annotation')
        for annotation_item in annotation_items:
            filename_item = annotation_item.find('./filename')
            text = filename_item.text

            if 'FigureSeparationTraining2016' in args.gt_path:
                image_path = os.path.join('/datasets/ImageCLEF/2016/training/FigureSeparationTraining2016/', text+'.jpg')
            elif 'FigureSeparationTest2016GT' in args.gt_path:
                image_path = os.path.join('/datasets/ImageCLEF/2016/test/FigureSeparationTest2016/', text+'.jpg')
            else:
                raise Exception('Error {0}'.format(args.gt_path))

            image_paths.append(image_path+"\n")

            image = read_image_bgr(image_path)
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            csv_path_individual = os.path.join('csv/', text + '.csv')
            jpg_path_individual = os.path.join('preview/', text + '.jpg')
            with open(csv_path_individual, 'w', newline='') as csvfile_individual:
                csv_writer_individual = csv.writer(csvfile_individual, delimiter=',')

                object_items = annotation_item.findall('./object')
                for idx, object_item in enumerate(object_items):
                    point_items = object_item.findall('./point')
                    x1 = point_items[0].get('x')
                    y1 = point_items[0].get('y')
                    x2 = point_items[3].get('x')
                    y2 = point_items[3].get('y')
                    if int(x1) >= int(x2) or int(y1) >= int(y2):
                        continue
                    csv_writer.writerow([image_path, x1, y1, x2, y2, 'panel'])
                    csv_writer_individual.writerow([image_path, x1, y1, x2, y2, 'panel'])

                    color = label_color(idx)
                    box = [int(x1), int(y1), int(x2), int(y2)]
                    draw_box(draw, box, color)

                cv2.imwrite(jpg_path_individual, draw)

    with open(args.list_path, "w") as text_file:
            text_file.writelines(image_paths)


if __name__ == '__main__':
    main()
