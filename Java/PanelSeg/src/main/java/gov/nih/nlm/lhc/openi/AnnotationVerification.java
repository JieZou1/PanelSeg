package gov.nih.nlm.lhc.openi;

import java.util.ArrayList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.w3c.dom.*;

import java.awt.Rectangle;
import java.io.*;
import java.nio.file.*;

public class AnnotationVerification 
{
	Path annotationFolder;
	ArrayList<Path> imagePaths;
	
	/**
	 * ctor, set annotationFolder and then collect all imagefiles and save in imagePaths
	 * @param annotationFolder
	 */
	AnnotationVerification(String annotationFolder)
	{
		this.annotationFolder = Paths.get(annotationFolder); 
		imagePaths = AlgorithmEx.CollectImageFiles(this.annotationFolder);
		System.out.println("Total number of image is: " + imagePaths.size());
	}

	/**
	 * Entry function for verify the annotation
	 */
	void verify()
	{
		for (int i = 0; i < imagePaths.size(); i++)
		{
			Path imagePath = imagePaths.get(i);
			String xmlFile = FilenameUtils.removeExtension(imagePath.toString()) + "_data.xml";

			//if (!xmlFile.endsWith("PMC3118333_1479-5876-9-71-2_data.xml")) continue;
			
			//The lists below will generate verification errors, but have manually checked to be Okay. 
			if (xmlFile.endsWith("PMC4363960_IJO-63-59-g004_data.xml")) continue;
			if (xmlFile.endsWith("PMC1193557_neh117f1_data.xml")) continue;
			if (xmlFile.endsWith("PMC2573880_1472-6882-8-55-7_data.xml")) continue;
			if (xmlFile.endsWith("PMC2996731_ijms-11-02839f5b_data.xml")) continue;
			if (xmlFile.endsWith("PMC3059462_1475-2875-10-S1-S4-2_data.xml")) continue;
			if (xmlFile.endsWith("PMC3274693_gr3_data.xml")) continue;
			if (xmlFile.endsWith("PMC3344238_ijms-13-04655f3b_data.xml")) continue;
			if (xmlFile.endsWith("PMC3327669_pone.0035368.g003_data.xml")) continue;
			if (xmlFile.endsWith("PMC3440350_pone.0044650.g003_data.xml")) continue;
			if (xmlFile.endsWith("PMC3440350_pone.0044650.g004_data.xml")) continue;
			if (xmlFile.endsWith("PMC3440350_pone.0044650.g005_data.xml")) continue;
			if (xmlFile.endsWith("PMC3525583_pone.0051481.g003_data.xml")) continue;
			if (xmlFile.endsWith("PMC3663038_sensors-08-01984f1b_data.xml")) continue;
			if (xmlFile.endsWith("PMC3591177_BMRI2013-737264.001_data.xml")) continue;
			if (xmlFile.endsWith("PMC3605171_1472-6882-13-56-15_data.xml")) continue;
			if (xmlFile.endsWith("PMC3744738_kcj-43-491-g003_data.xml")) continue;
			if (xmlFile.endsWith("PMC3796473_pone.0078217.g001_data.xml")) continue;
			if (xmlFile.endsWith("PMC4075677_AJP-2-196-g001_data.xml")) continue;
			if (xmlFile.endsWith("PMC4218991_12906_2014_1989_Fig1_HTML_data.xml")) continue;
			if (xmlFile.endsWith("PMC4264301_ASL-33-146-g003_data.xml")) continue;
			if (xmlFile.endsWith("PMC4389763_gfrr49_279_f7_data.xml")) continue;
			if (xmlFile.endsWith("PMC4408943_dddt-9-2285Fig5a_data.xml")) continue;
			if (xmlFile.endsWith("PMC4531038_ijn-10-5059Fig9a_data.xml")) continue;
			if (xmlFile.endsWith("PMC4531038_ijn-10-5059Fig10a_data.xml")) continue;
			if (xmlFile.endsWith("PMC4619947_ECAM2015-827472.004_data.xml")) continue;
			if (xmlFile.endsWith("PMC2148453_JCB29414.f5ef_data.xml")) continue;
			if (xmlFile.endsWith("PMC2667197_rsbl20080722f01_data.xml")) continue;
			if (xmlFile.endsWith("PMC3488839_97320630008953F1_data.xml")) continue;
			
			File annotationFile = new File(xmlFile);

			//Check whether annotation file exist or not.
			if (checkForMissingAnnotationFile(annotationFile)) continue;
			
			//Load annotation
			try {
				verifyPanelSegGt(annotationFile);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * Check whether the corresponding annotation XML exists or not.
	 * @param imagePath
	 * @return true if missing the annotation file.
	 */
	boolean checkForMissingAnnotationFile(File annotationFile)
	{
		if (!annotationFile.exists())
		{
			System.out.println("Missing Annotation File: " + annotationFile.toString());
			return true;
		}
		return false;
	}
	
	public static void main(String args[]) throws Exception
	{
		//Stop and print error msg if no arguments passed.
		if(args.length != 1)
		{
			System.out.println("Usage: java -cp PanelSeg.jar gov.nih.nlm.lhc.openi.AnnotationVerfication <annotation folder>");
			System.exit(0);
		}
		
		AnnotationVerification verification = new AnnotationVerification(args[0]);
		verification.verify();
	}

	static void verifyPanelSegGt(File gt_xml_file) throws Exception
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
				
				if (x < 0)	System.out.println("Negative x: " + text + ", in " + gt_xml_file +  "!");
				if (y < 0)	System.out.println("Negative y: " + text + ", in " + gt_xml_file +  "!");
				if (width < 0)	System.out.println("Negative width: " + text + ", in " + gt_xml_file +  "!");
				if (height < 0)	System.out.println("Negative height: " + text + ", in " + gt_xml_file +  "!");
				
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
				
				if (x < 0)	System.out.println("Negative x: " + text + ", in " + gt_xml_file +  "!");
				if (y < 0)	System.out.println("Negative y: " + text + ", in " + gt_xml_file +  "!");
				if (width < 0)	System.out.println("Negative width: " + text + ", in " + gt_xml_file +  "!");
				if (height < 0)	System.out.println("Negative height: " + text + ", in " + gt_xml_file +  "!");
				
				Rectangle labelRect = new Rectangle(x, y, width, height);
				labelRects.add(labelRect);
				String[] words = text.split("\\s+"); 
				labelNames.add(String.join(" ", ArrayUtils.remove(words, 0)));				
			}			
			else 
			{
				System.out.println("Unknown annotation, " + text + ", in " + gt_xml_file +  "!");
			}
		}

		//Check whether consistently having panel labels or not
		boolean noPanelLabels = panels.get(0).panelLabel.length() == 0;
		for (int i = 1; i < panels.size(); i++)
		{
			boolean noPanelLabelsCurr = panels.get(i).panelLabel.length() == 0;
			if (noPanelLabels != noPanelLabelsCurr)
			{
				System.out.println("Some labels may missing in: " + gt_xml_file +  "!");
			}
		}
		if (noPanelLabels)
		{
			if (labelRects.size() > 0 || labelNames.size() > 0)
				System.out.println("Extra Label boundingbox found in " + gt_xml_file +  "!");
			return;
		}
		
		//Reach here: Panel has label cases
		
		panels.sort(new PanelLabelAscending()); //sort the panels in ascending order

		//Check missing panels
		for (int i = 0; i < panels.size(); i++)
		{
			Panel panel = panels.get(i);
			String expectedPanelLabel = Character.toString((char)((int)('a' + i)));
			String panelLabel = panel.panelLabel.toLowerCase();
			if (!panelLabel.equals(expectedPanelLabel))
				System.out.println("Missing Panel " + expectedPanelLabel + ", in " + gt_xml_file +  "!");
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
				System.out.println("Label annotation is not found for Panel " + panel.panelLabel + ", in " + gt_xml_file +  "!");
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
				double interset_percentage, interset_percentage_max = Double.MIN_VALUE; int index_max = 0; int index;
				for (int j = 0; j < indexes.size(); j++)
				{
					index = indexes.get(j);
					Rectangle label_rect = labelRects.get(index);
					Rectangle intersect = panel_rect.intersection(label_rect);
					if (intersect.isEmpty()) interset_percentage = 0;
					else interset_percentage = ((double)(intersect.width*intersect.height) )/ ((double)(label_rect.width*label_rect.height));
					if (interset_percentage > interset_percentage_max)
					{
						index_max = index;
						interset_percentage_max = interset_percentage;
					}
				}
				panel.labelRect = labelRects.get(index_max);
				labelNames.remove(index_max);
				labelRects.remove(index_max);
			}
			
		}
		
		if (labelNames.size() > 0 || labelRects.size() > 0) //All elements in labelNames and labelRects should have been removed.
			System.out.println("Extra Label Boundingboxes found in " + gt_xml_file + "!");
		
	}
	
}
