package gov.nih.nlm.lhc.openi;

import java.util.ArrayList;
import java.io.IOException;
import java.nio.file.*;

public class AnnotationVerification 
{
	public static void main(String args[]) throws Exception
	{
		//Stop and print error msg if no agruments passed.
		if(args.length != 1){
			System.out.println("Usage: java -cp PanelSeg.jar gov.nih.nlm.ceb.openi.AnnotationVerfication <annotation folder>");
			System.exit(0);
		}
		
		ArrayList<String> gtXMLPaths = new ArrayList<String>();
		try (DirectoryStream<Path> dirStrm = Files.newDirectoryStream(Paths.get(args[0]))) 
		{			
			for (Path path : dirStrm)
			{
				String filename = path.toString();
				if (!filename.endsWith("_data.xml")) continue;
				gtXMLPaths.add(filename);
			}
		}
		catch (IOException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println("Total number of annotation is: " + gtXMLPaths.size());
		
		for (int i = 0; i < gtXMLPaths.size(); i++)
		{
			String file = gtXMLPaths.get(i);
			//if (!file.endsWith("PMC544880_1471-2369-5-18-2_data.xml")) continue;
			try
			{
				PanelSegEval.loadPanelSegGt(file);
			}
			catch (Exception e) 
			{
				System.out.println("Error: " + e.getMessage());
			}
		}
		
	}

}
