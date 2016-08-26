package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.lang3.ArrayUtils;
import org.w3c.dom.*;

import javax.xml.parsers.*;
import javax.xml.transform.*;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.awt.*;
import java.io.File;
import java.util.ArrayList;

/**
 * Utility functions related to iPhotoDraw annotations
 * Created by jzou on 8/26/2016.
 */
public class AnnotationiPhotoDraw
{
    /**
     * Load iPhotoDraw Annotations of panel segmentation
     * @param xml_file
     * @return
     * @throws Exception
     */
    static ArrayList<gov.nih.nlm.lhc.openi.panelseg.Panel> loadPanelSeg(File xml_file) throws Exception
    {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document doc = builder.parse(xml_file);

        ArrayList<gov.nih.nlm.lhc.openi.panelseg.Panel> panels = new ArrayList<>();
        ArrayList<Rectangle> labelRects = new ArrayList<>();
        ArrayList<String> labelNames = new ArrayList<>();

        NodeList shapeNodes = doc.getElementsByTagName("Shape");
        for (int i = 0; i < shapeNodes.getLength(); i++)
        {
            Node shapeNode = shapeNodes.item(i);
            Node blockTextNode = AlgorithmEx.getChildNode(shapeNode, "BlockText");
            Node textNode = AlgorithmEx.getChildNode(blockTextNode, "Text");
            String text = textNode.getTextContent().trim();
            String textLower = text.toLowerCase();

            if (textLower.startsWith("panel"))
            {	//It is a panel
                Node dataNode = AlgorithmEx.getChildNode(shapeNode, "Data");
                Node extentNode = AlgorithmEx.getChildNode(dataNode, "Extent");
                NamedNodeMap attributes = extentNode.getAttributes();
                int x = (int)(Double.parseDouble(attributes.getNamedItem("X").getTextContent()) + 0.5);
                int y = (int)(Double.parseDouble(attributes.getNamedItem("Y").getTextContent()) + 0.5);
                int width = (int)(Double.parseDouble(attributes.getNamedItem("Width").getTextContent()) + 0.5);
                int height = (int)(Double.parseDouble(attributes.getNamedItem("Height").getTextContent()) + 0.5);

                gov.nih.nlm.lhc.openi.panelseg.Panel panel = new gov.nih.nlm.lhc.openi.panelseg.Panel();
                panel.panelRect = new Rectangle(x, y, width, height);
                String[] words = text.split("\\s+");
                panel.panelLabel = String.join(" ", ArrayUtils.remove(words, 0));
                panels.add(panel);
            }
            else if (textLower.startsWith("label"))
            {	//It is a label
                Node dataNode = AlgorithmEx.getChildNode(shapeNode, "Data");
                Node extentNode = AlgorithmEx.getChildNode(dataNode, "Extent");
                NamedNodeMap attributes = extentNode.getAttributes();
                int x = (int)(Double.parseDouble(attributes.getNamedItem("X").getTextContent()) + 0.5);
                int y = (int)(Double.parseDouble(attributes.getNamedItem("Y").getTextContent()) + 0.5);
                int width = (int)(Double.parseDouble(attributes.getNamedItem("Width").getTextContent()) + 0.5);
                int height = (int)(Double.parseDouble(attributes.getNamedItem("Height").getTextContent()) + 0.5);

                Rectangle labelRect = new Rectangle(x, y, width, height);
                labelRects.add(labelRect);
                String[] words = text.split("\\s+");
                labelNames.add(String.join(" ", ArrayUtils.remove(words, 0)));
            }
            else
            {
                throw new Exception("Load Annotation Error: Unknown annotation, " + text + ", in " + xml_file +  "!");
            }
        }

        panels.sort(new PanelLabelAscending()); //sort the panels in ascending order of their labels

        if (labelRects.size() == 0) //All panels have no labels, so we don't need to match to panels.
        {
            //Make sure that panels are all labeled as "panel"
            for (int i = 0; i < panels.size(); i++)
            {
                gov.nih.nlm.lhc.openi.panelseg.Panel panel = panels.get(i);
                if (panel.panelLabel.length() > 0)
                    throw new Exception("Load Annotation Error: Unexpected annotation, " + panel.panelLabel + ", in " + xml_file +  "!");
            }
            return panels;
        }

        //Assign labels to panels
        for (int i = 0; i < panels.size(); i++)
        {
            gov.nih.nlm.lhc.openi.panelseg.Panel panel = panels.get(i);
            ArrayList<Integer> indexes = new ArrayList<Integer>();
            for (int j = 0; j < labelRects.size(); j++)
            {
                String label_name = labelNames.get(j);
                if (panel.panelLabel.toLowerCase().equals(label_name.toLowerCase()))	indexes.add(j);
            }

            if (indexes.size() == 0)
                throw new Exception("Load Annotation Error: Label annotation is not found for Panel " + panel.panelLabel + ", in " + xml_file +  "!");
            else if (indexes.size() == 1)
            {	//Only one label rect is found, perfect
                int index = indexes.get(0);
                panel.labelRect = labelRects.get(index);
                labelNames.remove(index);
                labelRects.remove(index);
            }
            else
            {	//More than 1 label rect is found, we assign the one, which is enclosed the most by the panel
                Rectangle panel_rect = panel.panelRect;
                double intersect_percentage, intersect_percentage_max = Double.MIN_VALUE; int index_max = 0; int index;
                for (int j = 0; j < indexes.size(); j++)
                {
                    index = indexes.get(j);
                    Rectangle label_rect = labelRects.get(index);
                    Rectangle intersect = panel_rect.intersection(label_rect);
                    if (intersect.isEmpty()) intersect_percentage = 0;
                    else intersect_percentage = ((double)(intersect.width*intersect.height) )/ ((double)(label_rect.width*label_rect.height));
                    if (intersect_percentage > intersect_percentage_max)
                    {
                        index_max = index;
                        intersect_percentage_max = intersect_percentage;
                    }
                }
                panel.labelRect = labelRects.get(index_max);
                labelNames.remove(index_max);
                labelRects.remove(index_max);
            }

        }

        if (labelNames.size() > 0 || labelRects.size() > 0) //All elements in labelNames and labelRects should have been removed.
            throw new Exception("Load Annotation Error: Extra Label Boundingboxes found in " + xml_file + "!");

        return panels;
    }

    private static Document createDocTemplate() throws ParserConfigurationException
    {
        // Create Document Template
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document doc = builder.newDocument();

        Element document_node = doc.createElement("Document");
        document_node.setAttribute("FileVersion", "1.0");

        Element image_options_node = doc.createElement("ImageOptions");
        image_options_node.setAttribute("Rotation", "0");
        image_options_node.setAttribute("IsNegative", "False");

        Element canvas_node = doc.createElement("Canvas");

        Element box_node = doc.createElement("Box");
        box_node.setAttribute("Height", "0");
        box_node.setAttribute("Width", "0");
        box_node.setAttribute("Top", "0");
        box_node.setAttribute("Left", "0");

        Element box_color_node = doc.createElement("BackColor");
        box_color_node.setAttribute("B", "255");
        box_color_node.setAttribute("G", "255");
        box_color_node.setAttribute("R", "255");
        box_color_node.setAttribute("Alpha", "255");

        Element flip_node = doc.createElement("Flip");
        flip_node.setAttribute("VerticalFlip", "False");
        flip_node.setAttribute("HorizontalFlip", "False");

        Element layers_node = doc.createElement("Layers");

        Element layer_node = doc.createElement("Layer");
        layer_node.setAttribute("Visible", "True");
        layer_node.setAttribute("Name", "Layer1");

        Element shapes_node = doc.createElement("Shapes");

        //HtmlTextNode line_break_node = doc.CreateTextNode("\n");
        //HtmlTextNode tab_node = doc.CreateTextNode("\t");

        //doc.DocumentNode.AppendChild(document_node);
        doc.appendChild(document_node);
        document_node.appendChild(image_options_node);
        document_node.appendChild(layers_node);


        layer_node.appendChild(shapes_node);
        layers_node.appendChild(layer_node);
        canvas_node.appendChild(box_node);
        canvas_node.appendChild(box_color_node);
        image_options_node.appendChild(canvas_node);
        image_options_node.appendChild(flip_node);

        return doc;
    }

    private static void AddRect(Document doc, String name, Rectangle rectangle)
    {
        Element shape_node = doc.createElement("Shape");
        shape_node.setAttribute("Type", "Rectangle");

        Element settings_node = doc.createElement("Settings");

        Element font_node = doc.createElement("Font");
        font_node.setAttribute("Name", "Arial");
        font_node.setAttribute("Style", "Regular");
        font_node.setAttribute("Size", "12");

        Element font_color_node = doc.createElement("Color");
        font_color_node.setAttribute("B", "0");
        font_color_node.setAttribute("G", "0");
        font_color_node.setAttribute("R", "0");
        font_color_node.setAttribute("Alpha", "255");

        Element line_node = doc.createElement("Line");
        line_node.setAttribute("Width", "1");
        line_node.setAttribute("OutlineType", "Color");
        line_node.setAttribute("Join", "Round");
        line_node.setAttribute("Dash", "Solid");

        Element line_color_node = doc.createElement("Color");
        if (name.startsWith("label"))
        {
            line_color_node.setAttribute("B", "234");
            line_color_node.setAttribute("G", "22");
            line_color_node.setAttribute("R", "30");
            line_color_node.setAttribute("Alpha", "255");
        }
        else
        {
            line_color_node.setAttribute("B", "30");
            line_color_node.setAttribute("G", "22");
            line_color_node.setAttribute("R", "234");
            line_color_node.setAttribute("Alpha", "255");
        }

        Element line_start_arrow_node = doc.createElement("StartArrowHead");
        line_start_arrow_node.setAttribute("Type", "None");
        line_start_arrow_node.setAttribute("WidthFactor", "2");
        line_start_arrow_node.setAttribute("HeightFactor", "1");

        Element line_end_arrow_node = doc.createElement("EndArrowHead");
        line_end_arrow_node.setAttribute("Type", "None");
        line_end_arrow_node.setAttribute("WidthFactor", "2");
        line_end_arrow_node.setAttribute("HeightFactor", "1");

        Element fill_node = doc.createElement("Fill");
        fill_node.setAttribute("FillType", "None");

        Element fill_color_node = doc.createElement("Color");
        fill_color_node.setAttribute("B", "255");
        fill_color_node.setAttribute("G", "255");
        fill_color_node.setAttribute("R", "255");
        fill_color_node.setAttribute("Alpha", "255");

        Element fill_gradient_settings_node = doc.createElement("GradientSettings");
        fill_gradient_settings_node.setAttribute("Type", "Linear");
        fill_gradient_settings_node.setAttribute("VerticalOffset", "0");
        fill_gradient_settings_node.setAttribute("HorizontalOffset", "0");
        fill_gradient_settings_node.setAttribute("Angle", "0");

        Element fill_gradient_starting_color_node = doc.createElement("StartingColor");
        fill_gradient_starting_color_node.setAttribute("B", "0");
        fill_gradient_starting_color_node.setAttribute("G", "0");
        fill_gradient_starting_color_node.setAttribute("R", "0");
        fill_gradient_starting_color_node.setAttribute("Alpha", "255");

        Element fill_gradient_ending_color_node = doc.createElement("EndingColor");
        fill_gradient_ending_color_node.setAttribute("B", "255");
        fill_gradient_ending_color_node.setAttribute("G", "255");
        fill_gradient_ending_color_node.setAttribute("R", "255");
        fill_gradient_ending_color_node.setAttribute("Alpha", "255");

        Element fill_gradient_blend_node = doc.createElement("Blend");

        Element fill_embedded_image_node = doc.createElement("EmbeddedImage");
        fill_embedded_image_node.setAttribute("Alpha", "255");
        fill_embedded_image_node.setAttribute("FileName", "");
        fill_embedded_image_node.setAttribute("ImageFillType", "Stretch");
        fill_embedded_image_node.setAttribute("Align", "Center");

        Element fill_embedded_image_stretch_setting_node = doc.createElement("StretchSettings");
        fill_embedded_image_stretch_setting_node.setAttribute("Type", "KeepOriginalSize");
        fill_embedded_image_stretch_setting_node.setAttribute("Align", "Center");
        fill_embedded_image_stretch_setting_node.setAttribute("ZoomFactor", "100");

        Element fill_embedded_image_stretch_setting_offset_node = doc.createElement("Offset");
        fill_embedded_image_stretch_setting_offset_node.setAttribute("Y", "0");
        fill_embedded_image_stretch_setting_offset_node.setAttribute("X", "0");

        Element fill_embedded_image_tile_setting_node = doc.createElement("TileSettings");
        fill_embedded_image_tile_setting_node.setAttribute("WrapMode", "Tile");

        Element fill_embedded_image_tile_setting_offset_node = doc.createElement("Offset");
        fill_embedded_image_tile_setting_offset_node.setAttribute("Y", "0");
        fill_embedded_image_tile_setting_offset_node.setAttribute("X", "0");

        Element fill_embedded_image_image_data_node = doc.createElement("ImageData");

        Element text_effect_node = doc.createElement("TextEffect");
        text_effect_node.setAttribute("UseTextEffect", "False");

        Element block_text_node = doc.createElement("BlockText");
        block_text_node.setAttribute("Align", "Center");
        block_text_node.setAttribute("RightToLeft", "No");

        Element text_node = doc.createElement("Text");
        Text text_value_node = doc.createTextNode(name);

        Element margin_node = doc.createElement("Margin");
        margin_node.setAttribute("Top", "0");
        margin_node.setAttribute("Left", "0");
        margin_node.setAttribute("Bottom", "0");
        margin_node.setAttribute("Right", "0");

        Element data_node = doc.createElement("Data");
        data_node.setAttribute("Rotation", "0");
        data_node.setAttribute("RoundCornerRadius", "0");
        data_node.setAttribute("IsRoundCorner", "False");

        Element extent_node = doc.createElement("Extent");
        extent_node.setAttribute("Height", Integer.toString(rectangle.height));
        extent_node.setAttribute("Width", Integer.toString(rectangle.width));
        extent_node.setAttribute("Y", Integer.toString(rectangle.y));
        extent_node.setAttribute("X", Integer.toString(rectangle.x));

        Element shapes_node = (Element)doc.getElementsByTagName("Shapes").item(0);
        shapes_node.appendChild(shape_node);
        shape_node.appendChild(settings_node);
        settings_node.appendChild(font_node);
        font_node.appendChild(font_color_node);
        settings_node.appendChild(line_node);
        line_node.appendChild(line_color_node);
        line_node.appendChild(line_start_arrow_node);
        line_node.appendChild(line_end_arrow_node);
        settings_node.appendChild(fill_node);
        fill_node.appendChild(fill_color_node);
        fill_node.appendChild(fill_gradient_settings_node);
        fill_gradient_settings_node.appendChild(fill_gradient_starting_color_node);
        fill_gradient_settings_node.appendChild(fill_gradient_ending_color_node);
        fill_gradient_settings_node.appendChild(fill_gradient_blend_node);
        fill_node.appendChild(fill_embedded_image_node);
        fill_embedded_image_node.appendChild(fill_embedded_image_stretch_setting_node);
        fill_embedded_image_stretch_setting_node.appendChild(fill_embedded_image_stretch_setting_offset_node);
        fill_embedded_image_node.appendChild(fill_embedded_image_tile_setting_node);
        fill_embedded_image_tile_setting_node.appendChild(fill_embedded_image_tile_setting_offset_node);
        fill_embedded_image_node.appendChild(fill_embedded_image_image_data_node);
        settings_node.appendChild(text_effect_node);
        shape_node.appendChild(block_text_node);
        block_text_node.appendChild(text_node);
        text_node.appendChild(text_value_node);
        block_text_node.appendChild(margin_node);
        shape_node.appendChild(data_node);
        data_node.appendChild(extent_node);

    }

    private static void saveDoc(Document doc, File xml_file) throws Exception
    {
        // write the content into xml file
        TransformerFactory transformerFactory = TransformerFactory.newInstance();
        javax.xml.transform.Transformer transformer = transformerFactory.newTransformer();
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");
        transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "2");
        DOMSource source = new DOMSource(doc);
        StreamResult result = new StreamResult(xml_file);
        transformer.transform(source, result);
    }

    static void savePanelSeg(File xml_file, ArrayList<Panel> panels) throws Exception
    {
        //Create document template
        Document doc = createDocTemplate();

        //Add panels into the template
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            if (panel.labelRect != null && !panel.labelRect.isEmpty())
                AddRect(doc, "label " + panel.panelLabel, panel.labelRect);
            if (panel.panelRect != null &&  !panel.panelRect.isEmpty())
                AddRect(doc, "panel " + panel.panelLabel, panel.panelRect);
        }

        //Save document into the XML file
        saveDoc(doc, xml_file);
    }


}
