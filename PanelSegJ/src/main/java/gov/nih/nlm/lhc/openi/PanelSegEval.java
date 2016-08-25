package gov.nih.nlm.lhc.openi;

import java.awt.Rectangle;
import java.io.*;
import java.util.*;
import org.w3c.dom.*;
import javax.xml.parsers.*;
import org.apache.commons.lang3.*;

/**
 * Evaluation related stuffs
 *
 * Created by jzou on 8/25/2016.
 */
public class PanelSegEval
{
    /**
     * Load Ground Truth Annotations of panel segmentation
     * @param gt_xml_file
     * @return
     * @throws Exception
     */
    static ArrayList<Panel> loadPanelSegGt(File gt_xml_file) throws Exception
    {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document doc = builder.parse(gt_xml_file);

        ArrayList<Panel> panels = new ArrayList<>();
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

                Panel panel = new Panel();
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
                throw new Exception("Load Annotation Error: Unknown annotation, " + text + ", in " + gt_xml_file +  "!");
            }
        }

        panels.sort(new PanelLabelAscending()); //sort the panels in ascending order of their labels

        if (labelRects.size() == 0) //All panels have no labels, so we don't need to match to panels.
        {
            //Make sure that panels are all labeled as "panel"
            for (int i = 0; i < panels.size(); i++)
            {
                Panel panel = panels.get(i);
                if (panel.panelLabel.length() > 0)
                    throw new Exception("Load Annotation Error: Unexpected annotation, " + panel.panelLabel + ", in " + gt_xml_file +  "!");
            }
            return panels;
        }

        //Assign labels to panels
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            ArrayList<Integer> indexes = new ArrayList<Integer>();
            for (int j = 0; j < labelRects.size(); j++)
            {
                String label_name = labelNames.get(j);
                if (panel.panelLabel.toLowerCase().equals(label_name.toLowerCase()))	indexes.add(j);
            }

            if (indexes.size() == 0)
                throw new Exception("Load Annotation Error: Label annotation is not found for Panel " + panel.panelLabel + ", in " + gt_xml_file +  "!");
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
            throw new Exception("Load Annotation Error: Extra Label Boundingboxes found in " + gt_xml_file + "!");

        return panels;
    }

    /**
     * Read the style annotation from the file
     */
    static HashMap<String, String> loadStyleMap(File styleFile)
    {
        HashMap<String, String> styles = new HashMap<String, String>();

        if(!styleFile.exists() || styleFile.isDirectory())
        {	//No styles have been marked yet
            System.out.println("Not able to find style.txt!");
            return null;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(styleFile)))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] words = line.split("\\s+");
                styles.put(words[0], words[1]);
            }
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return styles;
    }
}
