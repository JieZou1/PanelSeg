package gov.nih.nlm.lhc.openi.panelseg;

import java.io.*;
import java.nio.file.*;
import java.util.*;

import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_highgui;
import org.bytedeco.javacpp.opencv_imgcodecs;

/**
 * Created by jzou on 8/25/2016.
 */
final class DataVisualize extends Data
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 1)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar DataVisualize <annotation folder>");
            System.out.println("	This is a utility program to visualize through and modify the annotations.");
            System.exit(0);
        }

        DataVisualize visualizer = new DataVisualize(args[0]);
        visualizer.visualize();
    }

    /**
     * ctor, set setFolder and then collect all imagefiles and save in imagePaths
     * @param annotationFolder
     */
    private DataVisualize(String annotationFolder) throws Exception
    {
        super(annotationFolder);
    }

    /**
     * Run iPhotoDraw for manually modify the bounding box annotation
     * @param filepath
     */
    private void runiPhotoDraw(String filepath) throws Exception
    {
        Process process = new ProcessBuilder("C:\\Program Files (x86)\\iPhotoDraw\\iPhotoDraw.exe", filepath).start();
        process.waitFor();
    }

    /**
     * Entry function for verifying the annotation
     */
    private void visualize() throws Exception
    {
        if (styles == null)
        {   //No styles are loaded, we set all of them as "SINGLE"
            styles = new HashMap<>();
            for (int i = 0; i < imagePaths.size(); i++)
            {
                Path imagePath = imagePaths.get(i);
                String imageFile = imagePath.getFileName().toString();
                styles.put(imageFile, "SINGLE");
            }
            saveStyleMap(this.stylePath, this.styles);
        }

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
            try
            {
                panels = iPhotoDraw.loadPanelSegGt(annotationFile);
            }
            catch (Exception e) {
                System.out.println(e.getMessage());
                load_gt_error = true;
            }

            String key = imagePath.getFileName().toString();
            String style = styles.get(key);

            Mat img = opencv_imgcodecs.imread(imageFile);
            Mat imgAnnotated = load_gt_error ? img.clone() : Figure.drawAnnotation(img, panels, style);

            System.out.println();
            System.out.println(Integer.toString(i+1) +  ": Visualize Data for " + imageFile);
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
                opencv_highgui.imshow("Data", imgAnnotated);
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
                    try {	panels = iPhotoDraw.loadPanelSegGt(annotationFile);			}
                    catch (Exception e) {
                        System.out.println(e.getMessage());
                        load_gt_error = true;
                    }
                    imgAnnotated = load_gt_error ? img.clone() : Figure.drawAnnotation(img, panels, style);
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
                    saveStyleMap(this.stylePath, this.styles);
                    i++;
                    break;
                }
            }
        }
    }


}
