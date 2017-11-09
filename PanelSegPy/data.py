import argparse
import random
import shutil

import cv2
import os
import pandas as pd

import misc
from Config import Config
from Figure import Figure
from Panel import map_label, LABEL_MAPPING
from misc import print_progress_bar


def get_label_rpn_data(train_file_list):
    all_imgs = []
    classes_count = dict((c, 0) for c in LABEL_MAPPING.keys())
    visualise = False

    print('Read training annotation files')
    with open(train_file_list) as f:
        lines = f.readlines()

    # Remove whitespace characters, and then construct the figures
    figures = [Figure(line.strip()) for line in lines]

    print_progress_bar(0, len(figures), prefix='Progress:', suffix='Complete', length=50)
    for i, figure in enumerate(figures):
        figure.load_gt_annotation(which_annotation='label')
        figure.load_image()
        height, width = figure.image.shape[:2]
        annotation_data = {
            # 'imageset': 'trainval',
            'filepath': figure.image_path,
            'width': width,
            'height': height,
            'bboxes': []
        }

        for panel in figure.panels:
            class_name = panel.label
            if len(class_name) != 1:  # we handle single char label only
                continue

            # class_name = map_label(class_name)  # map to 50 classes
            class_name = 'fg'
            classes_count[class_name] += 1

            # We make all labels squares for now
            label_rect = misc.make_square(panel.label_rect)
            x1 = label_rect[0] + Figure.PADDING
            y1 = label_rect[1] + Figure.PADDING
            x2 = x1 + label_rect[2]
            y2 = y1 + label_rect[3]
            # difficulty = False
            annotation_data['bboxes'].append({
                'class': 'fg',
                'x1': x1,
                'x2': x2,
                'y1': y1,
                'y2': y2,
                # 'difficult': difficulty
            })
        if len(annotation_data['bboxes']) == 0:
            continue
        all_imgs.append(annotation_data)

        print_progress_bar(i + 1, len(figures), prefix='Progress:', suffix='Complete', length=50)

        if visualise:
            # img = cv2.imread(annotation_data['filepath'])
            img = figure.image
            for bbox in annotation_data['bboxes']:
                cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255))
            cv2.imshow('img', img)
            cv2.waitKey(0)

    return all_imgs, classes_count


def generate_statistics(c):
    """
    Generate label statistics for deciding some parameters of the algorithm
    :param c:
    :return:
    """
    list_file = c.list_file

    print('generate_statistics with list_file={0}'.format(list_file))
    input("Press Enter to continue...")

    with open(list_file) as f:
        lines = f.readlines()

    # Remove whitespace characters, and then construct the figures
    figures = [Figure(line.strip()) for line in lines]

    print_progress_bar(0, len(figures), prefix='Progress:', suffix='Complete', length=50)
    for i, figure in enumerate(figures):
        figure.load_gt_annotation(which_annotation='label')
        figure.load_image()
        print_progress_bar(i + 1, len(figures), prefix='Progress:', suffix='Complete', length=50)

    # Figure Image Statistics
    image_width, image_height = [], []
    for figure in figures:
        height, width = figure.image_orig.shape[:2]
        image_width.append(width)
        image_height.append(height)

    print('\nimage width statistics:')
    print(pd.Series(image_width).describe())
    print('\nimage height statistics:')
    print(pd.Series(image_height).describe())

    # Label Statistics
    label_width, label_height = [], []
    for figure in figures:
        for panel in figure.panels:
            width, height = panel.label_rect[2:]
            label_width.append(width)
            label_height.append(height)

    print('\nLabel width statistics:')
    print(pd.Series(label_width).describe())
    print('\nLabel height statistics:')
    print(pd.Series(label_height).describe())


def generate_label_train_samples(c):
    """
    Generate label train sample (image patches).
    :param c:
    :return:
    """
    list_file = c.list_file
    target_folder = c.labels_folder

    print('generate_label_train_samples with list_file={0} and save to {1}'.format(list_file, target_folder))
    input("Press Enter to continue...")

    with open(list_file) as f:
        lines = f.readlines()

    # Remove whitespace characters, and then construct the figures
    figures = [Figure(line.strip()) for line in lines]

    # Clear the folder

    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.mkdir(target_folder)

    for figure in figures:
        print("Processing {0}".format(figure.id))
        figure.load_gt_annotation(which_annotation='label')
        figure.load_image()
        if c.color_type == cv2.IMREAD_COLOR:
            figure.crop_label_patches(is_gray=False)
        else:
            figure.crop_label_patches(is_gray=True)
        figure.save_label_patches(target_folder)


def normalize_label_train_samples(c):
    src_folder = c.labels_folder
    dst_folder = c.labels_normalized_folder

    print('normalize_label_train_samples with src_folder={0} and target_folder={1}'.format(src_folder, dst_folder))
    input("Press Enter to continue...")

    # Collect only single char label only for now
    folders = [dI for dI in os.listdir(src_folder) if (',' not in dI) and os.path.isdir(os.path.join(src_folder, dI))]

    src_files = []
    for f in folders:
        folder = os.path.join(src_folder, f)
        src_file = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.png')]
        src_files.append(src_file)

    # print(pd.Series([len(k) for k in src_files]).describe())
    # select 5,000 if more than 5,000
    for i in range(len(src_files)):
        src_file = src_files[i]
        if len(src_file) > 5000:
            src_files[i] = random.sample(src_file, 5000)

    for src_file in src_files:
        for file in src_file:
            print("Processing {0}".format(file))
            if c.color_type == cv2.IMREAD_COLOR:
                src_img = cv2.imread(file, cv2.IMREAD_COLOR)
            else:
                src_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            dst_img = cv2.resize(src_img, (28, 28))
            path = file.replace(src_folder, dst_folder)
            folder, file = os.path.split(path)
            if not os.path.exists(folder):
                os.mkdir(folder)
            cv2.imwrite(path, dst_img)


def generate_nonlabel_train_samples(c):

    list_file = c.list_file
    target_folder = c.nonlabels_folder

    print('generate_nonlabel_train_samples with list_file={0} and target_folder={1}'.format(list_file, target_folder))
    input("Press Enter to continue...")

    with open(list_file) as f:
        lines = f.readlines()

    # Remove whitespace characters, and then construct the figures
    figures = [Figure(line.strip()) for line in lines]

    # Clear the folder
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    for figure in figures:
        print("Processing {0}".format(figure.id))
        figure.load_image()
        image_height, image_width = figure.image_orig.shape[:2]
        for i in range(50):
            x, y, s = random.randint(-5, image_width-5), random.randint(-5, image_height-5), round(random.gauss(20, 7))
            if (s < 5) or (s > 80):
                continue
            if (x + s - image_width > s / 2) or (0-x > s/2):
                continue
            if (y + s - image_height > s / 2) or (0-y > s/2):
                continue

            rect = (x, y, s, s)
            patch_file = figure.id + "_".join(str(x) for x in rect) + ".png"

            x += figure.PADDING
            y += figure.PADDING
            if c.color_type == cv2.IMREAD_COLOR:
                patch = figure.image[y:y+s, x:x+s]
            else:
                patch = figure.image_gray[y:y+s, x:x+s]

            # if patch.shape

            patch_file = os.path.join(target_folder, patch_file)
            cv2.imwrite(patch_file, patch)


def normalize_nonlabel_train_samples(c):
    src_folder = c.nonlabels_folder
    dst_folder = c.nonlabels_normalized_folder
    print('normalize_nonlabel_train_samples with src_folder={0} and target_folder={1}'.format(src_folder, dst_folder))
    input("Press Enter to continue...")

    src_files = [os.path.join(src_folder, file) for file in os.listdir(src_folder) if file.endswith('.png')]
    src_files = random.sample(src_files, 100000)
    # normalize images to 28x28
    for file in src_files:
        print("Processing {0}".format(file))
        if c.color_type == cv2.IMREAD_COLOR:
            src_img = cv2.imread(file, cv2.IMREAD_COLOR)
        else:
            src_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        dst_img = cv2.resize(src_img, (28, 28))
        path = file.replace(src_folder, dst_folder)
        folder, file = os.path.split(path)
        if not os.path.exists(folder):
            os.mkdir(folder)
        if not os.path.exists(path):
            cv2.imwrite(path, dst_img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prepare data for all kinds of training and testing')
    parser.add_argument('op',
                        help='an operation to be conducted',
                        type=str,
                        choices=[
                            'generate_statistics',

                            'generate_label_train_samples',
                            'normalize_label_train_samples',
                            'generate_nonlabel_train_samples',
                            'normalize_nonlabel_train_samples',
                                 ]
                        )
    # parser.add_argument('--root_folder',
    #                     help='the root folder of Exp',
    #                     type=str,
    #                     default='/Users/jie/projects/PanelSeg/ExpRcnn')

    args = parser.parse_args()

    configure = Config()

    if args.op == 'generate_statistics':
        generate_statistics(configure)

    elif args.op == 'generate_label_train_samples':
        generate_label_train_samples(configure)

    elif args.op == 'normalize_label_train_samples':
        normalize_label_train_samples(configure)

    elif args.op == 'generate_nonlabel_train_samples':
        generate_nonlabel_train_samples(configure)

    elif args.op == 'normalize_nonlabel_train_samples':
        normalize_nonlabel_train_samples(configure)

    else:
        print('Operation {0} is not implemented yet!'.format(args.op))




