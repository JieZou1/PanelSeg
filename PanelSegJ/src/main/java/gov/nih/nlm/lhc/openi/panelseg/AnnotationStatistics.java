package gov.nih.nlm.lhc.openi.panelseg;

import java.nio.file.Path;
import java.util.*;

/**
 * Created by jzou on 8/25/2016.
 */
public class AnnotationStatistics extends AnnotationAll
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 1)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar AnnotationStatistics <data folder>");
            System.exit(0);
        }

        AnnotationStatistics statistics = new AnnotationStatistics(args[0]);
        statistics.generateStatistics();
        System.out.println("Completed!");
    }

    /**
     * ctor, set dataFolder and then collect all annotationFolders
     * @param dataFolder
     */
    private AnnotationStatistics(String dataFolder)
    {
        super(dataFolder);
    }

    /**
     * Entry function for generating annotation statistics
     */
    private void generateStatistics()
    {
        int figureCount = 0, singleCount = 0, multiCount = 0; //Total figures, total single panel figures, total multi-panel figures.
        int panelCount = 0;
        HashMap<String, Integer> figures = new HashMap<>();

        for (Path annotation_folder : annotationFolders)
        {
            System.out.println("Check: " + annotation_folder);
            ArrayList<Path> imagePaths = AlgorithmEx.CollectImageFiles(annotation_folder);
            System.out.println("Total number of image is: " + imagePaths.size());
            ArrayList<Path> xmlPaths = AlgorithmEx.CollectXmlFiles(annotation_folder);
            System.out.println("Total number of XML files is: " + xmlPaths.size());

            int figure_count = 0, single_count = 0, multi_count = 0;
            int panel_count = 0;
            for (Path xml_path : xmlPaths)
            {
                String filename = xml_path.getFileName().toString();
                if (figures.containsKey(filename))
                {
                    int value = figures.get(filename);
                    figures.put(filename, value + 1);
                    System.out.println(filename + " is a duplicate.");
                    continue;
                }
                else figures.put(filename, 1);

                ArrayList<Panel> panels;
                try {
                    panels = AnnotationiPhotoDraw.loadPanelSeg(xml_path.toFile());
                } catch (Exception e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                    System.out.println("Load annotation error for " + filename);
                    continue;
                }

                //Passed all tests, the annotation is successfully loaded and the figure is unique.
                figure_count++;
                if (panels.size() == 1)	single_count++;
                else multi_count++;

                panel_count += panels.size();
            }
            System.out.println("Total number of figures is: " + figure_count);
            System.out.println("Total number of single-panel figures is: " + single_count);
            System.out.println("Total number of multi-panel figures is: " + multi_count);
            System.out.println("Total number of panels is: " + panel_count);
            System.out.println();

            figureCount += figure_count;
            singleCount += single_count;
            multiCount += multi_count;
            panelCount += panel_count;
        }

        //Statistics across all sets
        System.out.println("Overall Total number of figures is: " + figureCount);
        System.out.println("Overall Total number of single-panel figures is: " + singleCount);
        System.out.println("Overall Total number of multi-panel figures is: " + multiCount);
        System.out.println("Overall Total number of panels is: " + panelCount);


    }

}
