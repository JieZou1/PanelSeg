import random
import xml.etree.ElementTree as ET

import cv2
import os

import misc
from Panel import LABEL_CLASS_MAPPING, LABEL_ALL, Panel
from iPhotoDraw import save_annotation_xml


class Figure:
    """
    A class for a Figure
    image_path is the path to the figure image file
    id is the unique id to each figure
    image_orig is the original color image
    image is the extended (50 pixels in all directions) image
    image_gray is the grayscale image (extended)
    panels contain all panels
    """

    PADDING = 50

    def __init__(self, image_path):
        self.image_path = image_path
        path, self.file = os.path.split(image_path)
        path, folder = os.path.split(path)
        self.id = "{0}-{1}".format(folder, self.file)

        self.image_orig = None
        self.image = None
        self.image_gray = None
        self.image_width = 0
        self.image_height = 0

        self.panels = None

        # label detection results
        self.label_class_mapping = LABEL_CLASS_MAPPING
        self.class_label_mapping = {v: k for k, v in self.label_class_mapping.items()}
        self.label_prediction = None
        self.fg_labels = None
        self.fg_rois = None
        self.fg_scores = None

    def load_image(self):
        """
        Load the original image, and then padding the image and convert to gray scale
        :return:
        """
        self.image_orig = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        self.image = cv2.copyMakeBorder(
            self.image_orig, self.PADDING, self.PADDING, self.PADDING, self.PADDING, cv2.BORDER_CONSTANT, 0)
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image_height, self.image_width = self.image.shape[:2]

    def crop_patch(self, roi, is_gray=True):
        x, y, w, h = roi[0] + self.PADDING, roi[1] + self.PADDING, roi[2], roi[3]
        if is_gray:
            patch = self.image_gray[y:y+h, x:x+w]
        else:
            patch = self.image[y:y+h, x:x+w]
        return patch

    def crop_label_patches(self, is_gray=True):
        for panel in self.panels:
            # Collect all squares inside +20% and -10% bands
            square = misc.make_square(panel.label_rect)
            square_upper, square_lower = misc.extend_square(square, 0.5, 0.1)

            panel.label_rects = []
            for x in range(square_upper[0], square_lower[0]):
                for y in range(square_upper[1], square_lower[1]):
                    s_min = max([square_lower[0]+square_lower[2]-x, square_lower[1]+square_lower[3]-y])
                    s_max = min([square_upper[0]+square_upper[2]-x, square_upper[1]+square_upper[3]-y])
                    for s in range(s_min, s_max):
                        rect = [x, y, s, s]
                        panel.label_rects.append(rect)

            # We keep at most 50 patches
            panel.label_patches = []
            if len(panel.label_rects) > 50:
                panel.label_rects = random.sample(panel.label_rects, 50)
            for rect in panel.label_rects:
                patch = self.crop_patch(rect, is_gray)
                panel.label_patches.append(patch)

    def save_label_patches(self, target_folder):
        for panel in self.panels:
            for i in range(len(panel.label_patches)):
                # if len(panel.label) != 1:   # !!! We handle 1 letter label for now only!!!
                #     continue
                rect = panel.label_rects[i]
                patch = panel.label_patches[i]
                patch_file = self.id + "_".join(str(x) for x in rect) + ".png"
                folder = os.path.join(target_folder, str([ord(c) for c in panel.label]))
                if not os.path.exists(folder):
                    os.mkdir(folder)
                patch_file = os.path.join(folder, patch_file)
                cv2.imwrite(patch_file, patch)

    def load_gt_annotation(self,
                           which_annotation='label',  # label: label annotation only
                           ):
        """
        Load Ground Truth annotation
        :param which_annotation: 'label' load label annotation only; 'panel_and_label' load both panel and label annotation
        :return: None, the loaded annotation is saved to self.panels
        """
        iphotodraw_path = self.image_path.replace('.jpg', '_data.xml')
        self.load_annotation(iphotodraw_path, which_annotation)

    def load_annotation(self,
                        annotation_file_path,
                        which_annotation='label',  # label: label annotation only
                        file_type='iphotodraw'
                        ):
        """
        :param annotation_file_path: the file path to the annotation
        :param which_annotation: 'label' load label annotation only; 'panel_and_label' load both panel and label annotation
        :param file_type: 'iphotodraw' annotation file is in iphotodraw format
        :return: None, the loaded annotation is saved to self.panels
        """
        if which_annotation == 'label':
            if file_type == 'iphotodraw':
                self._load_annotation_label_iphotodraw(annotation_file_path)
            else:
                raise Exception(file_type + ' is unknown!')

        elif which_annotation == 'panel_and_label':
            if file_type == 'iphotodraw':
                self._load_annotation_panel_and_label_iphotodraw(annotation_file_path)
            else:
                raise Exception(file_type + ' is unknown!')

        else:
            raise Exception(which_annotation + ' is unknown!')

    def _load_annotation_label_iphotodraw(self, iphotodraw_path):
        """
        Load label annotation from iphotodraw formatted file
        :param iphotodraw_path:
        :return:
        """
        # create element tree object
        tree = ET.parse(iphotodraw_path)
        # get root element
        root = tree.getroot()

        shape_items = root.findall('./Layers/Layer/Shapes/Shape')

        # Read All Label Items
        label_items = []
        for shape_item in shape_items:
            text_item = shape_item.find('./BlockText/Text')
            text = text_item.text.lower()
            if text.startswith('label'):
                label_items.append(shape_item)

        # Form individual panels
        panels = []
        for label_item in label_items:
            text_item = label_item.find('./BlockText/Text')
            label = text_item.text
            words = label.split(' ')
            if len(words) is not 2:
                raise Exception(iphotodraw_path + ' ' + label + ' panel is not correct')
            label = words[1]

            extent_item = label_item.find('./Data/Extent')
            height = extent_item.get('Height')
            width = extent_item.get('Width')
            x = extent_item.get('X')
            y = extent_item.get('Y')
            label_rect = [round(float(x)), round(float(y)), round(float(width)), round(float(height))]

            panel = Panel(label, None, label_rect)
            panels.append(panel)

        self.panels = panels

    def _load_annotation_panel_and_label_iphotodraw(self, iphotodraw_path):
        """
        Load both panel and label annotation from iphotodraw formatted file
        :param iphotodraw_path:
        :return:
        """
        # create element tree object
        tree = ET.parse(iphotodraw_path)
        # get root element
        root = tree.getroot()

        shape_items = root.findall('./Layers/Layer/Shapes/Shape')

        # Read All Items (Panels and Labels)
        panel_items = []
        label_items = []
        for shape_item in shape_items:
            text_item = shape_item.find('./BlockText/Text')
            text = text_item.text.lower()
            if text.startswith('panel'):
                panel_items.append(shape_item)
            elif text.startswith('label'):
                label_items.append(shape_item)

        # Form individual panels
        panels = []
        for panel_item in panel_items:
            text_item = panel_item.find('./BlockText/Text')
            label = text_item.text
            words = label.split(' ')
            if len(words) is not 2:
                raise Exception(iphotodraw_path + ' ' + label + ' panel is not correct')
            label = words[1]

            extent_item = panel_item.find('./Data/Extent')
            height = extent_item.get('Height')
            width = extent_item.get('Width')
            x = extent_item.get('X')
            y = extent_item.get('Y')
            panel_rect = [round(float(x)), round(float(y)), round(float(width)), round(float(height))]

            panel = Panel(label, panel_rect, None)
            panels.append(panel)

        # Fill in label rects
        for panel in panels:
            for label_item in label_items:
                text_item = label_item.find('./BlockText/Text')
                label = text_item.text
                words = label.split(' ')
                if len(words) is not 2:
                    raise Exception(iphotodraw_path + ' ' + label + ' label is not correct')

                label = words[1]
                if label.lower() == panel.label.lower():
                    extent_item = label_item.find('./Data/Extent')
                    height = extent_item.get('Height')
                    width = extent_item.get('Width')
                    x = extent_item.get('X')
                    y = extent_item.get('Y')

                    label_rect = [round(float(x)), round(float(y)), round(float(width)), round(float(height))]
                    panel.label_rect = label_rect

        self.panels = panels

    def save_annotation(self,
                        annotation_folder,
                        which_annotation='label',  # label: label annotation only
                        file_type='iphotodraw'
                        ):
        """
        :param annotation_folder: the folder to save annotation, the file path is generated by id.
        :param which_annotation: 'label' save label annotation only; 'panel_and_label' save both panel and label annotation
        :param file_type: 'iphotodraw' annotation file is in iphotodraw format
        :return: None, the loaded annotation is saved to self.panels
        """
        if which_annotation == 'label':
            if file_type == 'iphotodraw':
                self._save_annotation_label_iphotodraw(annotation_folder)
            else:
                raise Exception(file_type + ' is unknown!')

        elif which_annotation == 'panel_and_label':
            if file_type == 'iphotodraw':
                # self._save_annotation_panel_and_label_iphotodraw(annotation_folder)
                raise Exception(file_type + ' is not implemented yet!')
            else:
                raise Exception(file_type + ' is unknown!')

        else:
            raise Exception(which_annotation + ' is unknown!')

    def _save_annotation_label_iphotodraw(self, annotation_folder):
        """
        save label annotation in iphotodraw formatted file
        :return:
        """
        # save original image
        img_path = os.path.join(annotation_folder, self.file)
        cv2.imwrite(img_path, self.image_orig)

        # save previews
        img = self.image.copy()
        labels = []
        if self.fg_rois is not None:
            color = (255, 0, 255)
            for i in range(len(self.fg_labels)):
                if self.fg_labels[i] == LABEL_ALL:
                    label = self.fg_labels[i]
                else:
                    label = self.class_label_mapping[int(self.fg_labels[i])]
                labels.append('label ' + label)
                prob = self.fg_scores[i]
                roi = self.fg_rois[i]
                pt1 = (roi[0], roi[1])
                pt2 = (roi[0] + roi[2], roi[1] + roi[3])
                cv2.rectangle(img, pt1, pt2, color)
                cv2.putText(img, label + '--{0}'.format(prob, '%.2f'), pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        preview_path = os.path.join(annotation_folder, self.file.replace('.jpg', '-preview.jpg'))
        cv2.imwrite(preview_path, img)

        # save annotation
        annotation_path = os.path.join(annotation_folder, self.file.replace('.jpg', '_data.xml'))
        save_annotation_xml(annotation_path, self.fg_rois, labels)

