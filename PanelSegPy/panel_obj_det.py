import tensorflow as tf
import numpy as np
import cv2
import time
import pickle
import os
from Figure import Figure

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('model_path', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\Panel\\faster-rcnn-resnet101_exported_graphs\\frozen_inference_graph.pb',
                    'the frozen trained model file')
flags.DEFINE_string('eval_path', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\eval.txt',
                    'the evaluation figure list file')
flags.DEFINE_string('result_folder', 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\Panel\\eval',
                    'the folder where the splitting results are saved to')
FLAGS = flags.FLAGS


class PanelSplitterObjDet(object):

    def __init__(self):
        PATH_TO_MODEL = FLAGS.model_path
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.detection_graph)

    def get_splitting_result(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})

        return boxes, scores, classes, num


def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou_rect(roi1, roi2):
    a = (roi1[0], roi1[1], roi1[0]+roi1[2], roi1[1] + roi1[3])
    b = (roi2[0], roi2[1], roi2[0]+roi2[2], roi1[1] + roi1[3])
    return iou(a, b)


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def load_image(image_path):
    image = cv2.imread(image_path)
    # image_resized = tf.cast(image_decoded, tf.float32)
    # image_resized = image_resized / 255.0
    return image


def is_duplicated_iou(panel_boxes, panel_box, iou_threshold):
    duplicated = False
    for k, p_box in enumerate(panel_boxes):
        iou = iou_rect(p_box, panel_box)
        if iou > iou_threshold:
            duplicated = True
    return duplicated


def panel_split(splitter, figure):
    result = splitter.get_splitting_result(figure.image)
    # pickle dump the result
    pickle_file = os.path.join(FLAGS.result_folder, figure.file.replace('.jpg', '.pickle'))
    with open(pickle_file, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    boxes, scores, classes, num = result
    # boxes[0, 1, 2, 3] is y_min, x_min, y_max, x_max

    # we process only one figure
    boxes = boxes[0]
    scores = scores[0]

    panel_boxes = []
    panel_scores = []
    for i, box in enumerate(boxes):
        if scores[i] < 0.95:
            break
        x0 = int(box[1]*figure.image_width + 0.5)
        y0 = int(box[0]*figure.image_height + 0.5)
        x1 = int(box[3]*figure.image_width + 0.5)
        y1 = int(box[2]*figure.image_height + 0.5)

        panel_box = [x0, y0, x1-x0, y1-y0]

        duplicated = is_duplicated_iou(panel_boxes, panel_box, iou_threshold=0.5)

        if not duplicated:
            panel_boxes.append(panel_box)
            panel_scores.append(scores[i])

    figure.panel_boxes = panel_boxes
    figure.panel_scores = panel_scores


def main(_):
    splitter = PanelSplitterObjDet()

    with open(FLAGS.eval_path) as f:
        lines = f.readlines()

    for idx, filepath in enumerate(lines):
        print(str(idx) + ': ' + filepath)
        filepath = filepath.strip()
        figure = Figure(filepath, padding=0)
        figure.load_image()

        st = time.time()
        panel_split(splitter, figure)
        print('Elapsed time = {}'.format(time.time() - st))

        #  save results
        figure.save_annotation(FLAGS.result_folder, 'panel')


if __name__ == '__main__':
    tf.app.run()
