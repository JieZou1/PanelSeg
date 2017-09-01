import xml.etree.ElementTree as ET


def load_iphoto_draw_annotation(annotation_file):
    # create element tree object
    tree = ET.parse(annotation_file)
    # get root element
    root = tree.getroot()

    shape_item = root.findall('./Layers/Layer/Shapes/Shape')

    for item in shape_item:
        label_item = item.find('./BlockText/Text')
        label = label_item.text
        pass

    pass


if __name__ == "__main__":
    annotation_file = "Z:/Users/jie/projects/PanelSeg/data/0/1465-9921-6-6-4_data.xml"
    load_iphoto_draw_annotation(annotation_file)
