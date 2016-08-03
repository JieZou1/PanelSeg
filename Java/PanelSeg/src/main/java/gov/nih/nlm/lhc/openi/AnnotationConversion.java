package gov.nih.nlm.lhc.openi;

import java.io.File;
import java.nio.file.*;
import java.util.ArrayList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class AnnotationConversion 
{
	private Path annotationFolder;
	private ArrayList<Path> annotationPaths;
	
	/**
	 * ctor, set annotationFolder and then collect all annotation file paths
	 * @param annotationFolder
	 */
	private AnnotationConversion(String annotationFolder)
	{
		this.annotationFolder = Paths.get(annotationFolder); 
		annotationPaths = AlgorithmEx.CollectXmlFiles(this.annotationFolder);
		System.out.println("Total number of XML files is: " + annotationPaths.size());
	}

	private void convert()
	{
		for (int i = 0; i < annotationPaths.size(); i++)
		{
			String xmlPath = annotationPaths.get(i).toString();
			//if (!xmlPath.endsWith("PMC3173667_kcj-41-464-g006_data.xml")) continue;
			
			File annotationFile = new File(xmlPath);
			//Load annotation
			try {
				convertPanelSegGt(annotationFile);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}		
		System.out.println("Completed!");
	}
	
	public static void main(String args[]) throws Exception
	{
		//Stop and print error msg if no arguments passed.
		if(args.length != 1)
		{
			System.out.println("Usage: java -cp PanelSeg.jar gov.nih.nlm.lhc.openi.AnnotationConversion <annotation folder>");
			System.exit(0);
		}
		
		AnnotationConversion conversion = new AnnotationConversion(args[0]);
		conversion.convert();
	}

	/**
	 * Fixing some annotation errors, including:
	 * 1. Rect width and height <= 0 
	 * 2. trim panel label texts.
	 * @param gt_xml_file
	 * @return
	 * @throws Exception
	 */
	static private void convertPanelSegGt(File gt_xml_file) throws Exception
	{
		String jpg_file = gt_xml_file.toString().replace("_data.xml", ".jpg");
		Mat jpg_image = opencv_imgcodecs.imread(jpg_file);
		int jpg_image_width = jpg_image.cols(); //int jpg_image_height = jpg_image.rows();

		String png_file = gt_xml_file.toString().replace("_data.xml", ".png");
		Mat png_image = opencv_imgcodecs.imread(png_file);
		int png_image_width = png_image.cols(); //int png_image_height = png_image.rows();
		
		double ratio = (double)jpg_image_width / (double)png_image_width;
		//double ratio_h = (double)jpg_image_height / (double)png_image_height;
		
		//if (jpg_image_width <= 512) return;	//if the original image width is less than 512, the image is not resized, so we don't need to change bounding box annotations.
		
		DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
		DocumentBuilder builder = factory.newDocumentBuilder();
		Document doc = builder.parse(gt_xml_file);

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

				x *= ratio; xNode.setTextContent(Double.toString(x));
				y *= ratio; yNode.setTextContent(Double.toString(y));
				w *= ratio; wNode.setTextContent(Double.toString(w));
				h *= ratio; hNode.setTextContent(Double.toString(h));
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
