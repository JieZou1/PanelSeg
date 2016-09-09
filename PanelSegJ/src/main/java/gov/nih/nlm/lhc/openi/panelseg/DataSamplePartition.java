package gov.nih.nlm.lhc.openi.panelseg;

import java.io.*;
import java.nio.file.Path;
import java.util.*;

/**
 * Created by jzou on 8/25/2016.
 */
public class DataSamplePartition extends DataAll
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 1)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar DataSamplePartition");
            System.out.println("    This utility program partitions all samples into training and evaluation set");
            System.out.println("    Evaluation set contains 1000 randomly selected sample and the remaining are training set");
            System.exit(0);
        }

        String dataFolder = "\\Users\\jie\\projects\\PanelSeg\\data\\";
        DataSamplePartition miscSamplePartition = new DataSamplePartition(dataFolder);
        miscSamplePartition.partition();
        System.out.println("Completed!");
    }

    private DataSamplePartition(String dataFolder)
    {
        super(dataFolder);
    }

    private void partition()
    {
        List<Path> all = new ArrayList<>();

        //Collect all samples
        for (int i = 0; i < setFolders.size(); i++)
        {
            Path folder = setFolders.get(i);
            List images = AlgMiscEx.collectImageFiles(folder);
            all.addAll(images);
        }
        //Save all sample list
        writeList("all.txt", all);

        //Randomly select 1000 as evaluation set
        long seed = 1; int eval_size = 1000;
        Collections.shuffle(all, new Random(seed));
        writeList("allshuffled.txt", all);

        List<Path> eval = all.subList(0, eval_size);
        List<Path> train = all.subList(eval_size, all.size());

        writeList("eval.txt", eval);
        writeList("train.txt", train);
    }

    private void writeList(String filename, List<Path> list)
    {
        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename))))
        {
            for (Path path : list)
            {
                bw.write(path.toString());
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
}
