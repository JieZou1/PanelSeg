package gov.nih.nlm.lhc.openi;

import java.awt.Rectangle;
import java.util.ArrayList;
import org.w3c.dom.*;
import javax.xml.parsers.*;
import org.apache.commons.lang3.*;

public class PanelSegEval 
{
	/**
	 * Load Ground Truth Annotations of panel segmentation
	 * @param xml_file
	 * @return
	 * @throws Exception
	 */
	static ArrayList<Panel> loadPanelSegGt(String gt_xml_file) throws Exception
	{
		DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
		DocumentBuilder builder = factory.newDocumentBuilder();
		Document doc = builder.parse(gt_xml_file);
		
		ArrayList<Panel> panels = new ArrayList<Panel>();
		ArrayList<Rectangle> labelRects = new ArrayList<Rectangle>();
		ArrayList<String> labelNames = new ArrayList<String>();
		
		NodeList shapeNodes = doc.getElementsByTagName("Shape");
		
		for (int i = 0; i < shapeNodes.getLength(); i++)
		{
			Node shapeNode = shapeNodes.item(i);
			Node blockTextNode = AlgorithmEx.getChildNode(shapeNode, "BlockText");
			Node textNode = AlgorithmEx.getChildNode(blockTextNode, "Text");
			String text = textNode.getTextContent().toLowerCase();
			
			if (text.startsWith("panel"))
			{	//It is a panel
				Node dataNode = AlgorithmEx.getChildNode(shapeNode, "Data");
				Node extentNode = AlgorithmEx.getChildNode(dataNode, "Extent");
				NamedNodeMap attributes = extentNode.getAttributes();
				int x = (int)(Double.parseDouble(attributes.getNamedItem("X").getTextContent()));
				int y = (int)(Double.parseDouble(attributes.getNamedItem("Y").getTextContent()));
				int width = (int)(Double.parseDouble(attributes.getNamedItem("Width").getTextContent()));
				int height = (int)(Double.parseDouble(attributes.getNamedItem("Height").getTextContent()));
				
				Panel panel = new Panel();
				panel.panelRect = new Rectangle(x, y, width + 1, height + 1); //Looks like that iPhotoDraw uses [] for range instead of [)
				String[] words = text.split("\\s+");
				panel.panelLabel = String.join(" ", ArrayUtils.remove(words, 0));
				panels.add(panel);
			}
			else if (text.startsWith("label"))
			{	//It is a label
				Node dataNode = AlgorithmEx.getChildNode(shapeNode, "Data");
				Node extentNode = AlgorithmEx.getChildNode(dataNode, "Extent");
				NamedNodeMap attributes = extentNode.getAttributes();
				int x = (int)(Double.parseDouble(attributes.getNamedItem("X").getTextContent()));
				int y = (int)(Double.parseDouble(attributes.getNamedItem("Y").getTextContent()));
				int width = (int)(Double.parseDouble(attributes.getNamedItem("Width").getTextContent()));
				int height = (int)(Double.parseDouble(attributes.getNamedItem("Height").getTextContent()));
			
				Rectangle labelRect = new Rectangle(x, y, width + 1, height + 1); //Looks like that iPhotoDraw uses [] for range instead of [)
				labelRects.add(labelRect);
				String[] words = text.split("\\s+"); 
				labelNames.add(String.join(" ", ArrayUtils.remove(words, 0)));				
			}			
			else 
			{
				throw new Exception("Load Annotation Error: Unknown annotation, " + text + ", in " + gt_xml_file +  "!");
			}
		}
		
		//Match labels to panels
		for (int i = 0; i < labelRects.size(); i++)
		{
			String labelName = labelNames.get(i);			
			boolean found = false;
			for (int j = 0; j < panels.size(); j++)
			{
				String panelName = panels.get(j).panelLabel;
				if (labelName.equals(panelName))
				{
					panels.get(j).labelRect = labelRects.get(i);
					found = true;
					break;
				}
			}
			
			if (found)	continue;
			
			throw new Exception("Load Annotation Error: Not able to find matching Panel for Label " + labelName + " in " + gt_xml_file + "!");
			
			//Not found by matching labels, we check with intersections
//			Rectangle labelRect = labelRects.get(i);
//			for (int j = 0; j < panels.size(); j++)
//			{
//				Rectangle panelRect = panels.get(j).panelRect;
//				if (panelRect.intersects(labelRect))
//				{
//					panels.get(j).labelRect = labelRect;
//					panels.get(j).panelLabel = labelName;
//					found = true;
//					break;
//				}
//			}
//			
//			if (found) continue;
//			
//			//Not found by matching labels, and checking intersections, we check with union
//			int min_area = Integer.MAX_VALUE; int min_j = -1;
//			for (int j = 0; j < panels.size(); j++)
//			{
//				Rectangle panelRect = panels.get(j).panelRect;
//				Rectangle union = panelRect.union(labelRect);
//				int area = union.width * union.height;
//				if (area < min_area)
//				{
//					min_area = area; min_j = j;
//				}
//			}
//			panels.get(min_j).labelRect = labelRect;
//			panels.get(min_j).panelLabel = labelName;
		}
		
		return panels;
	}

}
