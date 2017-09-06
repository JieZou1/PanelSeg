import xml.etree.ElementTree as ET

from Panel import Panel


class Figure:
    '''
    A class for a Figure
    image_file_path can serve as a unique id to each figure
    '''

    def __init__(self, image_path):
        self.image_path = image_path
        self.panels = []

    def load_annotation_iphotodraw(self, iphotodraw_path):
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
            panel_rect = (x, y, width, height)

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

                    label_rect = (x, y, width, height)
                    panel.label_rect = label_rect

        self.panels = panels

    def load_gt_annotation_iphotodraw(self):
        iphotodraw_path = self.image_path.replace('.jpg', '_data.xml')
        self.load_annotation_iphotodraw(iphotodraw_path)
        return self.panels
