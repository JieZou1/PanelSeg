import tensorflow as tf
from Figure import Figure
import os
import pickle

flags = tf.app.flags
flags.DEFINE_string('eval_list', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\eval.txt',
                    'the evaluation file list')
flags.DEFINE_string('auto_dir', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\Panel\\eval',
                    'the automated results')
flags.DEFINE_string('eval_file', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\Panel\\eval.txt',
                    'the evaluation result')
FLAGS = flags.FLAGS


def intersect(rect0, rect1):
    ai = [rect0[0], rect0[1], rect0[0]+rect0[2], rect0[1]+rect0[3]]
    bi = [rect1[0], rect1[1], rect1[0]+rect1[2], rect1[1]+rect1[3]]
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return None
    return [x, y, w, h]


def eval_ImageCLEF():

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
                intersection_area = intersection[2] * intersection[3]
                auto_area = auto_panel.panel_rect[2] * auto_panel.panel_rect[3]
                overlapping = intersection_area / auto_area
                if overlapping > max_overlapping:
                    max_overlapping = overlapping
                    max_auto_idx = auto_idx
            if max_overlapping > 0.66:
                correct += 1
                picked_auto_idx[max_auto_idx] = True

        k = max(len(auto_panels), len(gt_panels))
        accuracy = correct / k
        recall = correct / len(gt_panels)
        if len(auto_panels) == 0:
            precision = 0
        else:
            precision = correct / len(auto_panels)
        return accuracy, recall, precision

    #  Read all figures to be evaluated
    with open(FLAGS.eval_list) as f:
        lines = f.readlines()

    with open(FLAGS.eval_file, 'a') as f:

        overall_accuracy = 0
        overall_precision = 0
        overall_recall = 0

        for idx, filepath in enumerate(lines):
            print(str(idx) + ': ' + filepath)
            filepath = filepath.strip()
            figure = Figure(filepath, padding=0)

            figure.load_gt_annotation(which_annotation='panel')
            gt_panels = figure.panels

            auto_file = os.path.join(FLAGS.auto_dir, figure.file.replace('.jpg', '_data.xml'))
            figure.load_annotation(auto_file, which_annotation='panel')
            auto_panels = figure.panels

            accuracy, recall, precision = compute_metric(gt_panels, auto_panels)
            overall_accuracy += accuracy
            overall_precision += precision
            overall_recall += recall

            f.write("{0}: {1}\t{2}\t{3}\n".format(figure.id, accuracy, precision, recall))

        overall_accuracy /= len(lines)
        overall_recall /= len(lines)
        overall_precision /= len(lines)

        f.write("Overall Accuracy: {}\n".format(overall_accuracy))
        f.write("Overall Recall: {}\n".format(overall_recall))
        f.write("Overall Precision: {}\n".format(overall_precision))


def eval_mAP():
    #  Read all figures to be evaluated
    with open(FLAGS.eval_list) as f:
        lines = f.readlines()

    all_boxes = []
    all_scores = []
    all_num = []
    figures = []
    with open(FLAGS.eval_file, 'a') as f:

        for idx, filepath in enumerate(lines):
            print(str(idx) + ': ' + filepath)
            filepath = filepath.strip()
            figure = Figure(filepath, padding=0)
            figure.load_gt_annotation(which_annotation='panel')

            pickle_file = os.path.join(FLAGS.auto_dir, figure.file.replace('.jpg', '.pickle'))
            with open(pickle_file, 'rb') as handle:
                boxes, scores, classes, num = pickle.load(handle)

            all_boxes.append(boxes[0])
            all_scores.append(scores[0])
            all_num.append(num[0])
            figures.append(figure)

    # TODO: mAP evaluation


def main(_):
    # eval_ImageCLEF()
    eval_mAP()


if __name__ == '__main__':
    tf.app.run()
