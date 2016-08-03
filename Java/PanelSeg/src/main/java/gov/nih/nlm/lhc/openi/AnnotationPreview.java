package gov.nih.nlm.lhc.openi;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Map;

import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Scalar;

public class AnnotationPreview 
{
	private Path annotationFolder;
	private ArrayList<Path> imagePaths;
	private Map<String, String> styles;
	
	/**
	 * ctor, set annotationFolder and then collect all imagefiles and save in imagePaths
	 * @param annotationFolder
	 */
	private AnnotationPreview(String annotationFolder)
	{
		this.annotationFolder = Paths.get(annotationFolder); 
		imagePaths = AlgorithmEx.CollectImageFiles(this.annotationFolder);
		System.out.println("Total number of image is: " + imagePaths.size());

		File styleFile = new File(annotationFolder, "style.txt"); 
		styles = PanelSegEval.loadStyleMap(styleFile);
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
	
	/**
	 * Entry function for generating annotation preview
	 */
	private void generatePreview()
	{
		for (int i = 0; i < imagePaths.size(); i++)
		{
			Path imagePath = imagePaths.get(i);
			String imageFile = imagePath.toString();
			String xmlFile = FilenameUtils.removeExtension(imageFile) + "_data.xml";
			
			//if (!xmlFile.endsWith("PMC3025345_kcj-40-684-g002_data.xml")) {i++; continue; }
			
			//Load annotation
			File annotationFile = new File(xmlFile);
			ArrayList<Panel> panels = null; boolean load_gt_error = false;
			try {	panels = PanelSegEval.loadPanelSegGt(annotationFile);			} 
			catch (Exception e) {
				System.out.println(e.getMessage());
				load_gt_error = true;
			}
			
			System.out.println(Integer.toString(i+1) +  ": Generate Annotation Preview for " + imageFile);
			
			Mat img = opencv_imgcodecs.imread(imageFile);
			String key = imagePath.getFileName().toString();
			String style = styles.get(key);
			Mat imgAnnotated = load_gt_error ? img.clone() : drawAnnotation(img, panels, style);
			
			Path parent = imagePath.getParent();
			parent = parent.resolve("preview");
			parent = parent.resolve(imagePath.getFileName().toString());
			Path preview_path = imagePath.getParent().resolve("preview").resolve(imagePath.getFileName().toString());
			opencv_imgcodecs.imwrite(preview_path.toString(), imgAnnotated);
		}
	}
	
	public static void main(String args[]) throws Exception
	{
		//Stop and print error msg if no arguments passed.
		if(args.length != 1)
		{
			System.out.println("Usage: java -cp PanelSeg.jar gov.nih.nlm.lhc.openi.AnnotationPreview <annotation folder>");
			System.exit(0);
		}
		
		AnnotationPreview preview = new AnnotationPreview(args[0]);
		preview.generatePreview();
		System.out.println("Completed!");
	}
}
