package gov.nih.nlm.lhc.openi.panelseg;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * Created by jzou on 8/25/2016.
 */
public class AnnotationDuplicate
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 1)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar AnnotationDuplicate");
            System.out.println("    This utility program checks the annotation folders to find duplicated figures");
            System.out.println("    Since the Figures are downloaded based on keywords search in several sets, duplication is possible");
            System.exit(0);
        }

        String dataFolder = "\\Users\\jie\\projects\\PanelSeg\\data\\";
        AnnotationDuplicate duplicate = new AnnotationDuplicate(dataFolder);
        duplicate.findDuplicate();
        System.out.println("Completed!");
    }

    private Path dataFolder;
    private ArrayList<Path> annotationFolders;



    private AnnotationDuplicate(String dataFolder)
    {
        this.dataFolder = Paths.get(dataFolder);
        annotationFolders = AlgorithmEx.CollectSubfolders(this.dataFolder);
    }

    private void findDuplicate()
    {
        long seed = 1;

        for (int i = 0; i < annotationFolders.size(); i++)
        {
            Path folder = annotationFolders.get(i);
            ArrayList<Path> images = AlgorithmEx.CollectImageFiles(folder);
            Collections.shuffle(annotationFolders, new Random(seed));
        }
    }
}
