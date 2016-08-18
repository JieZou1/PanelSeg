package gov.nih.nlm.lhc.openi;

import java.io.*;
import java.nio.file.*;
import java.util.*;

import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_highgui;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;

public class AnnotationVisualize 
{
	private Path annotationFolder;
	private ArrayList<Path> imagePaths;
	private File styleFile;
	private Map<String, String> styles;
	
	/**
	 * ctor, set annotationFolder and then collect all imagefiles and save in imagePaths
	 * @param annotationFolder
	 */
	private AnnotationVisualize(String annotationFolder)
	{
		this.annotationFolder = Paths.get(annotationFolder); 
		imagePaths = AlgorithmEx.CollectImageFiles(this.annotationFolder);
		System.out.println("Total number of image is: " + imagePaths.size());

		styleFile = new File(annotationFolder, "style.txt"); 
		styles = PanelSegEval.loadStyleMap(styleFile);
	}
	
	/**
	 * Save the style annotation into the file
	 */
	private void saveStyleMap()
	{
		try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(styleFile))))
		{
			for (Map.Entry<String, String> entry : styles.entrySet()) 
			{
			    String key = entry.getKey();
			    String value = entry.getValue();
				bw.write(key + "\t " + value);
				bw.newLine();
			}	    
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * Draw annotation onto the image for visualization purpose
	 * @param img
	 * @param panels
	 * @return
	 */
	private Mat drawAnnotation(Mat img, ArrayList<Panel> panels, String style)
	{
		Mat imgAnnotated = new Mat(); 
		opencv_core.copyMakeBorder(img, imgAnnotated, 0, 100, 0, 50, opencv_core.BORDER_CONSTANT, new Scalar());
		
		//Draw bounding box first
		for (int i = 0; i < panels.size(); i++)
		{
			Panel panel = panels.get(i);
			Scalar scalar = AlgorithmEx.getColor(i);

			Rect panel_rect = AlgorithmEx.Rectangle2Rect(panel.panelRect); 
			opencv_imgproc.rectangle(imgAnnotated, panel_rect, scalar, 3, 8, 0);

			if (panel.panelLabel.length() != 0)
			{
				Rect label_rect = AlgorithmEx.Rectangle2Rect(panel.labelRect);
				opencv_imgproc.rectangle(imgAnnotated, label_rect, scalar, 1, 8, 0);
			}
		}
		
		//Draw labels to make the text stand out.
		for (int i = 0; i < panels.size(); i++)
		{
			Panel panel = panels.get(i);
			Scalar scalar = AlgorithmEx.getColor(i);

			if (panel.panelLabel.length() != 0)
			{
				Rect label_rect = AlgorithmEx.Rectangle2Rect(panel.labelRect);
				Point bottom_left = new Point(label_rect.x() + label_rect.width(), label_rect.y() + label_rect.height() + 50);
				opencv_imgproc.putText(imgAnnotated, panel.panelLabel, bottom_left, opencv_imgproc.CV_FONT_HERSHEY_PLAIN, 5, scalar, 3, 8, false);
			}
		}

		{//Draw Style Annotation
			Scalar scalar = AlgorithmEx.getColor(1);
			Point bottom_left = new Point(0, img.rows() + 100);
			opencv_imgproc.putText(imgAnnotated, style, bottom_left, opencv_imgproc.CV_FONT_HERSHEY_PLAIN, 2, scalar, 3, 8, false);
		}
		
		return imgAnnotated;
	}
	
	private void runiPhotoDraw(String filepath)
	{
		try {
			Process process = new ProcessBuilder("C:\\Program Files (x86)\\iPhotoDraw\\iPhotoDraw.exe", filepath).start();
			process.waitFor();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * Entry function for verify the annotation
	 */
	private void visualize()
	{
		for (int i = 0; i < imagePaths.size();)
		{
			if (i < 0 ) i = 0;	if (i >= imagePaths.size()) i = imagePaths.size()-1;

			Path imagePath = imagePaths.get(i);
			String imageFile = imagePath.toString();
			String xmlFile = FilenameUtils.removeExtension(imageFile) + "_data.xml";
			
			//if (!xmlFile.endsWith("PMC3083004_ZooKeys-072-023-g010_data.xml")) {i++; continue; }
			
			//Load annotation
			File annotationFile = new File(xmlFile);
			ArrayList<Panel> panels = null; boolean load_gt_error = false;
			try {	panels = PanelSegEval.loadPanelSegGt(annotationFile);			} 
			catch (Exception e) {
				System.out.println(e.getMessage());
				load_gt_error = true;
			}
			
			String key = imagePath.getFileName().toString();
			String style = styles.get(key);
			
			Mat img = opencv_imgcodecs.imread(imageFile);
			Mat imgAnnotated = load_gt_error ? img.clone() : drawAnnotation(img, panels, style);

			System.out.println();
			System.out.println(Integer.toString(i+1) +  ": Visualize Annotation for " + imageFile);
			System.out.println("The style is " + style);
			
			System.out.println("Press n, go to the next figure");
			System.out.println("Press p, go to the previous figure");
			System.out.println("Press c, run iPhotoDraw to manually modify the annotation");
			System.out.println("Press 1, mark it as SINGLE-PANEL");
			System.out.println("Press 2, mark it as MULTI-PANEL");
			System.out.println("Press 3, mark it as STITCHED-MULTI-PANEL");
			
			while (true)
			{
				opencv_highgui.imshow("Image", img);
				opencv_highgui.imshow("Annotation", imgAnnotated);
				int c;
				while (true)
				{
					c = opencv_highgui.waitKey();
					if (c == (int)'n' || c == (int)'p' || c == (int)'c' || c == (int)'1' || c == (int)'2' || c == (int)'3')	break;
				}
				
				if (c == (int)'n')	{	i++; break;	}		//Move to the next one
				else if (c == (int)'p')	{	i--; break;	}	//Move to the previous one
				else if (c == (int)'c')						//Run iPhotoDraw to manually modify the annotation
				{
					runiPhotoDraw(imageFile);
					load_gt_error = false;
					try {	panels = PanelSegEval.loadPanelSegGt(annotationFile);			} 
					catch (Exception e) {
						System.out.println(e.getMessage());
						load_gt_error = true;
					}
					imgAnnotated = load_gt_error ? img.clone() : drawAnnotation(img, panels, style);
				}
				else if (c == (int)'1' || c == (int)'2' || c == (int)'3')
				{
					if (c == (int)'1')
					{
						String value = "SINGLE";
						styles.put(key, value);
						System.out.println(key + " is Marked as SINGLE-PANEL");
					}
					else if (c == (int)'2')
					{
						String value = "MULTI";
						styles.put(key, value);
						System.out.println(key + " is Marked as MULTI-PANEL");
					}
					else if (c == (int)'3')
					{
						String value = "STITCHED";
						styles.put(key, value);
						System.out.println(key + " is Marked as STITCHED-MULTI-PANEL");
					}
					saveStyleMap();
					i++;
					break;
				}
			}
		}
	}
	
	public static void main(String args[]) throws Exception
	{
		//Stop and print error msg if no arguments passed.
		if(args.length != 1)
		{
			System.out.println("Usage: java -cp PanelSeg.jar gov.nih.nlm.lhc.openi.AnnotationVisualize <annotation folder>");
			System.out.println("	This is a utility program to visualize through and modify the annotations.");
			System.exit(0);
		}
		
		AnnotationVisualize visualizer = new AnnotationVisualize(args[0]);
		visualizer.visualize();
	}

}
