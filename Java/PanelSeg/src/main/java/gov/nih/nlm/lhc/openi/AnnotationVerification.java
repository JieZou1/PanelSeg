package gov.nih.nlm.lhc.openi;

import java.util.ArrayList;

import org.apache.commons.io.FilenameUtils;

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
			File annotationFile = new File(FilenameUtils.removeExtension(imagePath.toString()) + "_data.xml");
//			if (!annotationFile.endsWith("PMC544880_1471-2369-5-18-2_data.xml")) continue;

			//Check whether annotation file exist or not.
			if (checkForMissingAnnotationFile(annotationFile)) continue;
			
			//Load annotation
			ArrayList<Panel> panels = null;
			try {
				panels = PanelSegEval.loadPanelSegGt(annotationFile);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			//Check for panels which have empty Rectangle
			if (checkForEmptyRect(panels)) continue;
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
	
	boolean checkForEmptyRect(ArrayList<Panel> panels)
	{
		
		
		return true;
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
}
