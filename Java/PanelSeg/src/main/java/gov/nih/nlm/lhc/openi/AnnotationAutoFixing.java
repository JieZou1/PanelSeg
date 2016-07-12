package gov.nih.nlm.lhc.openi;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

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
				PanelSegEval.fixPanelSegGt(annotationFile);
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

}
