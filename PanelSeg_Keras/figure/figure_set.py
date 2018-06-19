import os
import logging
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from figure import misc
from figure.panel_seg_error import PanelSegError
from figure.figure import Figure


class FigureSet:
    """
    A class for a FigureSet
    """

    def __init__(self):
        self.list_file = None
        self.files = None

    def load_list(self, list_file):
        self.list_file = list_file
        self.files = misc.read_sample_list(list_file)

    def validate_annotation(self):
        for idx, file in enumerate(self.files):
            # if idx != 5:
            #     continue
            logging.info('Validate Annotation of Image %d: %s.', idx, file)

            try:
                if not os.path.exists(file):
                    raise PanelSegError('Could not find %s.'.format(file))
                figure = Figure()
                figure.load_image(file)

                xml_path = os.path.join(figure.image_path.replace('.jpg', '_data.xml'))
                if not os.path.exists(xml_path):
                    raise PanelSegError('Could not find %s.'.format(xml_path))
                figure.load_annotation_iphotodraw(xml_path)

            except PanelSegError as ex:
                logging.warning(ex.message)
                continue

    def save_gt_preview(self):
        for idx, file in enumerate(self.files):
            # if idx != 5:
            #     continue
            logging.info('Generate GT Annotation Preview for Image %d: %s.', idx, file)

            try:
                if not os.path.exists(file):
                    raise PanelSegError('Could not find %s.'.format(file))
                figure = Figure()
                figure.load_image(file)

                xml_path = os.path.join(figure.image_path.replace('.jpg', '_data.xml'))
                if not os.path.exists(xml_path):
                    raise PanelSegError('Could not find %s.'.format(xml_path))
                figure.load_annotation_iphotodraw(xml_path)

            except PanelSegError as ex:
                logging.warning(ex.message)
                continue

            # Save preview
            folder, file = os.path.split(figure.image_path)
            folder = os.path.join(folder, "prev")
            figure.save_preview(folder)

    def convert_to_csv(self):
        pass

