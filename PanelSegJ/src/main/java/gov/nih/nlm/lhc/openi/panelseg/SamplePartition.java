package gov.nih.nlm.lhc.openi.panelseg;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Created by jzou on 8/25/2016.
 */
public class SamplePartition
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 1)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar SamplePartition");
            System.out.println("    This utility program partitions all samples into training and evaluation set");
            System.out.println("    Evaluation set contains 1000 randomly selected sample and the remaining are training set");
            System.exit(0);
        }

        String dataFolder = "\\Users\\jie\\projects\\PanelSeg\\data\\";
        SamplePartition samplePartition = new SamplePartition(dataFolder);
        samplePartition.Partition();
        System.out.println("Completed!");
    }

    private Path dataFolder;
    private ArrayList<Path> annotationFolders;



    private SamplePartition(String dataFolder)
    {
        this.dataFolder = Paths.get(dataFolder);
        annotationFolders = AlgorithmEx.CollectSubfolders(this.dataFolder);
    }

    private void Partition()
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
