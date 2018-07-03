import os
import pickle
import sys
import argparse
import tensorflow as tf

from figure.figure import Figure


def parse_args(args):
    parser = argparse.ArgumentParser(description='Evaluate Panel Split.')

    parser.add_argument('--eval_list', help='Path to the eval list file.',
                        # default='/Users/jie/projects/PanelSeg/ExpKeras/clef_eval.txt'
                        default='/Users/jie/projects/PanelSeg/ExpKeras/eval.txt'
                        )
    parser.add_argument('--auto_dir', help='The folder to the splitting result.',
                        default='/Users/jie/projects/PanelSeg/ExpKeras/panel_split/retinanet/panelseg/VGG19/results'
                        # default='/Users/jie/projects/PanelSeg/ExpKeras/panel_split/rentinanet/train-with-clef2016/results-imageclef2016-resnet152_csv_8-0.05-0.44-0.66.h5'
                        # default='/Users/jie/projects/PanelSeg/ExpKeras/panel_split/rentinanet/train-with-ours/results-ours-resnet50_csv_50.h5'
                        )
    parser.add_argument('--eval_file', help='The evaluation result file.',
                        default='eval.txt')

    return parser.parse_args(args)


def intersect(rect0, rect1):
    # ai = [rect0[0], rect0[1], rect0[0]+rect0[2], rect0[1]+rect0[3]]
    # bi = [rect1[0], rect1[1], rect1[0]+rect1[2], rect1[1]+rect1[3]]
    x = max(rect0[0], rect1[0])
    y = max(rect0[1], rect1[1])
    w = min(rect0[2], rect1[2]) - x
    h = min(rect0[3], rect1[3]) - y
    if w < 0 or h < 0:
        return None
    return [x, y, x+w, y+h]


def eval_ImageCLEF(args):

    def compute_metric(gt_panels, auto_panels):

        correct = 0
        picked_auto_idx = [False] * len(auto_panels)
        for gt_idx, gt_panel in enumerate(gt_panels):
            max_overlapping = -1
            max_auto_idx = -1
            for auto_idx, auto_panel in enumerate(auto_panels):
                if picked_auto_idx[auto_idx]:
                    continue
                intersection = intersect(auto_panel.panel_rect, gt_panel.panel_rect)
                if intersection is None:
                    continue
                intersection_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
                auto_area = (auto_panel.panel_rect[2] - auto_panel.panel_rect[0]) * (auto_panel.panel_rect[3] - auto_panel.panel_rect[1])
                overlapping = intersection_area / auto_area
                if overlapping > max_overlapping:
                    max_overlapping = overlapping
                    max_auto_idx = auto_idx
            if max_overlapping > 0.66:
            # if max_overlapping > 0.44:
                correct += 1
                picked_auto_idx[max_auto_idx] = True

        return correct

    #  Read all figures to be evaluated
    with open(args.eval_list) as f:
        lines = f.readlines()

    with open(args.eval_file, 'a') as f:

        overall_correct_count = 0
        overall_gt_count = 0
        overall_auto_count = 0

        overall_accuracy = 0.0
        overall_recall = 0.0
        overall_precision = 0.0

        for idx, filepath in enumerate(lines):
            # if '1471-213X-8-30-5' not in filepath:
            #     continue

            print(str(idx) + ': ' + filepath)
            filepath = filepath.strip()
            figure = Figure(filepath)
            if 'clef_eval' in args.eval_list:
                csv_path = os.path.join(figure.image_path.replace('.jpg', '.csv'))
                figure.load_annotation_csv(csv_path)
            else:
                figure.load_image()
                xml_path = os.path.join(figure.image_path.replace('.jpg', '_data.xml'))
                figure.load_annotation_iphotodraw(xml_path)
            gt_panels = figure.panels

            image_folder, image_file = os.path.split(figure.image_path)
            auto_file = os.path.join(args.auto_dir, image_file.replace('.jpg', '.csv'))
            figure.load_annotation_csv(auto_file)
            auto_panels = figure.panels

            correct = compute_metric(gt_panels, auto_panels)

            k = max(len(auto_panels), len(gt_panels))
            accuracy = correct / k
            recall = correct / len(gt_panels)
            if len(auto_panels) == 0:
                precision = 0
            else:
                precision = correct / len(auto_panels)

            f.write("{0}: {1}\t{2}\t{3}\n".format(figure.image_path, accuracy, precision, recall))

            overall_accuracy += accuracy
            overall_recall += recall
            overall_precision += precision

            overall_correct_count += correct
            overall_gt_count += len(gt_panels)
            overall_auto_count += len(auto_panels)

        overall_accuracy /= len(lines)
        overall_precision /= len(lines)
        overall_recall /= len(lines)

        f.write("Overall ImageCLEF Accuracy: {}, Precision: {}, Recall: {}\n".format(
            overall_accuracy, overall_precision, overall_recall))

        overall_accuracy = overall_correct_count / max(overall_auto_count, overall_gt_count)
        overall_recall = overall_correct_count / overall_gt_count
        overall_precision = overall_correct_count / overall_auto_count

        f.write("Overall Accuracy: {}, Precision: {}, Recall: {}\n".format(
            overall_accuracy, overall_precision, overall_recall))


def eval_mAP(args):
    #  Read all figures to be evaluated
    with open(args.eval_list) as f:
        lines = f.readlines()

    all_boxes = []
    all_scores = []
    all_num = []
    figures = []
    with open(args.eval_file, 'a') as f:

        for idx, filepath in enumerate(lines):
            print(str(idx) + ': ' + filepath)
            filepath = filepath.strip()
            figure = Figure(filepath, padding=0)
            figure.load_gt_annotation(which_annotation='panel')

            pickle_file = os.path.join(args.auto_dir, figure.file.replace('.jpg', '.pickle'))
            with open(pickle_file, 'rb') as handle:
                boxes, scores, classes, num = pickle.load(handle)

            all_boxes.append(boxes[0])
            all_scores.append(scores[0])
            all_num.append(num[0])
            figures.append(figure)

    # TODO: mAP evaluation


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    print('Evaluation list file is {0}'.format(args.eval_list))
    print('Auto splitting folder is {0}'.format(args.auto_dir))

    eval_ImageCLEF(args)


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()
