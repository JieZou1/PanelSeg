package gov.nih.nlm.lhc.openi;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

import javax.xml.parsers.*;
import javax.xml.transform.*;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.w3c.dom.*;

public class AnnotationAutoFixing 
{
	Path annotationFolder;
	ArrayList<Path> annotationPaths;
	
	/**
	 * ctor, set annotationFolder and then collect all imagefiles and save in imagePaths
	 * @param annotationFolder
	 */
	AnnotationAutoFixing(String annotationFolder)
	{
		this.annotationFolder = Paths.get(annotationFolder); 
		annotationPaths = AlgorithmEx.CollectXmlFiles(this.annotationFolder);
		System.out.println("Total number of XML files is: " + annotationPaths.size());
	}
	
	/**
	 * Entry function for fixing the annotation
	 */
	void Fix()
	{
		for (int i = 0; i < annotationPaths.size(); i++)
		{
			String xmlPath = annotationPaths.get(i).toString();
			//if (!xmlPath.endsWith("PMC3173667_kcj-41-464-g006_data.xml")) continue;
			
			File annotationFile = new File(xmlPath);
			//Load annotation
			try {
				fixPanelSegGt(annotationFile);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	
	public static void main(String args[]) throws Exception
	{
		//Stop and print error msg if no arguments passed.
		if(args.length != 1)
		{
			System.out.println("Usage: java -cp PanelSeg.jar gov.nih.nlm.lhc.openi.AnnotationAutoFixing <annotation folder>");
			System.exit(0);
		}
		
		AnnotationAutoFixing fixing = new AnnotationAutoFixing(args[0]);
		fixing.Fix();
	}

	/**
	 * Fixing some annotation errors, including:
	 * 1. Rect width and height <= 0 
	 * 2. trim panel label texts.
	 * @param gt_xml_file
	 * @return
	 * @throws Exception
	 */
	static void fixPanelSegGt(File gt_xml_file) throws Exception
	{
		String imageFile = gt_xml_file.toString().replace("_data.xml", ".png");
		Mat image = opencv_imgcodecs.imread(imageFile);
		
		DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
		DocumentBuilder builder = factory.newDocumentBuilder();
		Document doc = builder.parse(gt_xml_file);

		Node shapesNode = doc.getElementsByTagName("Shapes").item(0);
		NodeList shapeNodes = doc.getElementsByTagName("Shape");
		
		for (int i = shapeNodes.getLength() - 1; i >= 0; i--)
		{
			Node shapeNode = shapeNodes.item(i);
			Node blockTextNode = AlgorithmEx.getChildNode(shapeNode, "BlockText");
			Node textNode = AlgorithmEx.getChildNode(blockTextNode, "Text");
			String text = textNode.getTextContent().trim();
			textNode.setTextContent(text);
			
			String textLower = text.toLowerCase();
			
			if (textLower.startsWith("panel") || textLower.startsWith("label"))
			{	//It is a panel
				Node dataNode = AlgorithmEx.getChildNode(shapeNode, "Data");
				Node extentNode = AlgorithmEx.getChildNode(dataNode, "Extent");
				NamedNodeMap attributes = extentNode.getAttributes();
				Node xNode = attributes.getNamedItem("X");
				Node yNode = attributes.getNamedItem("Y");
				Node wNode = attributes.getNamedItem("Width"); 
				Node hNode = attributes.getNamedItem("Height");
				double x = Double.parseDouble(xNode.getTextContent());
				double y = Double.parseDouble(yNode.getTextContent());
				double w = Double.parseDouble(wNode.getTextContent());
				double h = Double.parseDouble(hNode.getTextContent());
				
				if (w <= 0 || h <= 0)
				{
					System.out.println("Bounding Box Removed: " + text + ", in " + gt_xml_file +  "!");
					shapesNode.removeChild(shapeNode);
					continue;
				}
				if (x < 0)
				{
					w += x; /*We keep the original right*/ 		x = 0;
					xNode.setTextContent(Double.toString(x)); wNode.setTextContent(Double.toString(w));
					System.out.println("Negative X Modified: " + text + ", in " + gt_xml_file +  "!");
				}
				if (y < 0)
				{
					h += y; /*We keep the original bottom */	y = 0;
					yNode.setTextContent(Double.toString(y)); hNode.setTextContent(Double.toString(h));
					System.out.println("Negative Y Modified: " + text + ", in " + gt_xml_file +  "!");
				}
				if (x + w > image.cols())
				{
					w = image.cols() - x;
					wNode.setTextContent(Double.toString(w));
					System.out.println("Outside Right Modified: " + text + ", in " + gt_xml_file +  "!");
				}
				if (y + h > image.rows())
				{
					h = image.rows() - y;
					hNode.setTextContent(Double.toString(h));
					System.out.println("Outside Bottom Modified: " + text + ", in " + gt_xml_file +  "!");
				}
			}
			else 
			{
				throw new Exception("Load Annotation Error: Unknown annotation, " + text + ", in " + gt_xml_file +  "!");
			}
		}

		// write the content into xml file
	    TransformerFactory transformerFactory = TransformerFactory.newInstance();
	    javax.xml.transform.Transformer transformer = transformerFactory.newTransformer();
	    transformer.setOutputProperty(OutputKeys.INDENT, "yes");
	    transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "2");			
	    DOMSource source = new DOMSource(doc);
		StreamResult result = new StreamResult(gt_xml_file);
		transformer.transform(source, result);
	}
	
}
