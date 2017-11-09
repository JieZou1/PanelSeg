import xml.etree.ElementTree as ET

import Figure


def save_annotation_xml(annotation_path, rois, labels):
    document_node, shapes_node = create_doc_template()

    for i in range(len(rois)):
        roi = rois[i]
        label = labels[i]
        add_roi(shapes_node, roi, label)

    tree = ET.ElementTree(document_node)
    tree.write(annotation_path)


def create_doc_template():
    document_node = ET.Element('Document')
    document_node.set("FileVersion", "1.0")
    image_options_node = ET.Element("ImageOptions")
    image_options_node.set("Rotation", "0")
    image_options_node.set("IsNegative", "False")

    canvas_node = ET.Element("Canvas")

    box_node = ET.Element("Box")
    box_node.set("Height", "0")
    box_node.set("Width", "0")
    box_node.set("Top", "0")
    box_node.set("Left", "0")

    box_color_node = ET.Element("BackColor")
    box_color_node.set("B", "255")
    box_color_node.set("G", "255")
    box_color_node.set("R", "255")
    box_color_node.set("Alpha", "255")

    flip_node = ET.Element("Flip")
    flip_node.set("VerticalFlip", "False")
    flip_node.set("HorizontalFlip", "False")

    layers_node = ET.Element("Layers")

    layer_node = ET.Element("Layer")
    layer_node.set("Visible", "True")
    layer_node.set("Name", "Layer1")

    shapes_node = ET.Element("Shapes")

    document_node.append(image_options_node)
    document_node.append(layers_node)

    layer_node.append(shapes_node)
    layers_node.append(layer_node)
    canvas_node.append(box_node)
    canvas_node.append(box_color_node)
    image_options_node.append(canvas_node)
    image_options_node.append(flip_node)

    return document_node, shapes_node


def add_roi(shapes_node, rectangle, name):

    shape_node = ET.Element("Shape")
    shape_node.set("Type", "Rectangle")

    settings_node = ET.Element("Settings")

    font_node = ET.Element("Font")
    font_node.set("Name", "Arial")
    font_node.set("Style", "Regular")
    font_node.set("Size", "12")

    font_color_node = ET.Element("Color")
    font_color_node.set("B", "0")
    font_color_node.set("G", "0")
    font_color_node.set("R", "0")
    font_color_node.set("Alpha", "255")

    line_node = ET.Element("Line")
    line_node.set("Width", "1")
    line_node.set("OutlineType", "Color")
    line_node.set("Join", "Round")
    line_node.set("Dash", "Solid")

    line_color_node = ET.Element("Color")
    if name.startswith("label"):
        line_color_node.set("B", "234")
        line_color_node.set("G", "22")
        line_color_node.set("R", "30")
        line_color_node.set("Alpha", "255")
    else:
        line_color_node.set("B", "30")
        line_color_node.set("G", "22")
        line_color_node.set("R", "234")
        line_color_node.set("Alpha", "255")

    line_start_arrow_node = ET.Element("StartArrowHead")
    line_start_arrow_node.set("Type", "None")
    line_start_arrow_node.set("WidthFactor", "2")
    line_start_arrow_node.set("HeightFactor", "1")

    line_end_arrow_node = ET.Element("EndArrowHead")
    line_end_arrow_node.set("Type", "None")
    line_end_arrow_node.set("WidthFactor", "2")
    line_end_arrow_node.set("HeightFactor", "1")

    fill_node = ET.Element("Fill")
    fill_node.set("FillType", "None")

    fill_color_node = ET.Element("Color")
    fill_color_node.set("B", "255")
    fill_color_node.set("G", "255")
    fill_color_node.set("R", "255")
    fill_color_node.set("Alpha", "255")

    fill_gradient_settings_node = ET.Element("GradientSettings")
    fill_gradient_settings_node.set("Type", "Linear")
    fill_gradient_settings_node.set("VerticalOffset", "0")
    fill_gradient_settings_node.set("HorizontalOffset", "0")
    fill_gradient_settings_node.set("Angle", "0")

    fill_gradient_starting_color_node = ET.Element("StartingColor")
    fill_gradient_starting_color_node.set("B", "0")
    fill_gradient_starting_color_node.set("G", "0")
    fill_gradient_starting_color_node.set("R", "0")
    fill_gradient_starting_color_node.set("Alpha", "255")

    fill_gradient_ending_color_node = ET.Element("EndingColor")
    fill_gradient_ending_color_node.set("B", "255")
    fill_gradient_ending_color_node.set("G", "255")
    fill_gradient_ending_color_node.set("R", "255")
    fill_gradient_ending_color_node.set("Alpha", "255")

    fill_gradient_blend_node = ET.Element("Blend")

    fill_embedded_image_node = ET.Element("EmbeddedImage")
    fill_embedded_image_node.set("Alpha", "255")
    fill_embedded_image_node.set("FileName", "")
    fill_embedded_image_node.set("ImageFillType", "Stretch")
    fill_embedded_image_node.set("Align", "Center")

    fill_embedded_image_stretch_setting_node = ET.Element("StretchSettings")
    fill_embedded_image_stretch_setting_node.set("Type", "KeepOriginalSize")
    fill_embedded_image_stretch_setting_node.set("Align", "Center")
    fill_embedded_image_stretch_setting_node.set("ZoomFactor", "100")

    fill_embedded_image_stretch_setting_offset_node = ET.Element("Offset")
    fill_embedded_image_stretch_setting_offset_node.set("Y", "0")
    fill_embedded_image_stretch_setting_offset_node.set("X", "0")

    fill_embedded_image_tile_setting_node = ET.Element("TileSettings")
    fill_embedded_image_tile_setting_node.set("WrapMode", "Tile")

    fill_embedded_image_tile_setting_offset_node = ET.Element("Offset")
    fill_embedded_image_tile_setting_offset_node.set("Y", "0")
    fill_embedded_image_tile_setting_offset_node.set("X", "0")

    fill_embedded_image_image_data_node = ET.Element("ImageData")

    text_effect_node = ET.Element("TextEffect")
    text_effect_node.set("UseTextEffect", "False")

    block_text_node = ET.Element("BlockText")
    block_text_node.set("Align", "Center")
    block_text_node.set("RightToLeft", "No")

    # text_node = ET.Element("Text").text = name

    margin_node = ET.Element("Margin")
    margin_node.set("Top", "0")
    margin_node.set("Left", "0")
    margin_node.set("Bottom", "0")
    margin_node.set("Right", "0")

    data_node = ET.Element("Data")
    data_node.set("Rotation", "0")
    data_node.set("RoundCornerRadius", "0")
    data_node.set("IsRoundCorner", "False")

    extent_node = ET.Element("Extent")
    extent_node.set("Height", str(rectangle[3]))
    extent_node.set("Width", str(rectangle[2]))
    extent_node.set("Y", str(rectangle[1] - Figure.Figure.PADDING))
    extent_node.set("X", str(rectangle[0] - Figure.Figure.PADDING))

    shapes_node.append(shape_node)
    shape_node.append(settings_node)
    settings_node.append(font_node)
    font_node.append(font_color_node)
    settings_node.append(line_node)
    line_node.append(line_color_node)
    line_node.append(line_start_arrow_node)
    line_node.append(line_end_arrow_node)
    settings_node.append(fill_node)
    fill_node.append(fill_color_node)
    fill_node.append(fill_gradient_settings_node)
    fill_gradient_settings_node.append(fill_gradient_starting_color_node)
    fill_gradient_settings_node.append(fill_gradient_ending_color_node)
    fill_gradient_settings_node.append(fill_gradient_blend_node)
    fill_node.append(fill_embedded_image_node)
    fill_embedded_image_node.append(fill_embedded_image_stretch_setting_node)
    fill_embedded_image_stretch_setting_node.append(fill_embedded_image_stretch_setting_offset_node)
    fill_embedded_image_node.append(fill_embedded_image_tile_setting_node)
    fill_embedded_image_tile_setting_node.append(fill_embedded_image_tile_setting_offset_node)
    fill_embedded_image_node.append(fill_embedded_image_image_data_node)
    settings_node.append(text_effect_node)
    shape_node.append(block_text_node)
    # block_text_node.append(text_node)
    ET.SubElement(block_text_node, 'Text').text = name
    block_text_node.append(margin_node)
    shape_node.append(data_node)
    data_node.append(extent_node)
