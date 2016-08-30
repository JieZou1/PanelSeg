package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FileUtils;

import java.io.*;
import java.lang.reflect.Array;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Created by jzou on 8/25/2016.
 */
public class AnnotationDuplicate extends AnnotationAll
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 1)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar AnnotationDuplicate");
            System.out.println("    This utility program checks the annotation folders to find all duplicated figures");
            System.out.println("    Since the Figures are downloaded based on keywords search in several sets, duplication is possible");
            System.exit(0);
        }

        String dataFolder = "\\Users\\jie\\projects\\PanelSeg\\data\\";
        AnnotationDuplicate duplicate = new AnnotationDuplicate(dataFolder);
        duplicate.findDuplicate();
        System.out.println("Completed!");
    }

    private AnnotationDuplicate(String dataFolder)
    {
        super(dataFolder);
    }

    private void findDuplicate()
    {
        HashMap<String, ArrayList<Path>> collection = new HashMap<>();

        for (int i = 0; i < annotationFolders.size(); i++)
        {
            Path folder = annotationFolders.get(i);
            ArrayList<Path> images = AlgorithmEx.CollectImageFiles(folder);

            for (int j = 0; j < images.size(); j++)
            {
                Path path = images.get(j);
                String filename = path.getFileName().toString();

                ArrayList<Path> value = new ArrayList<>();
                if (collection.containsKey(filename)) value = collection.get(filename);
                value.add(path);
                collection.put(filename, value);
            }
        }

        //Output duplicates
        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("duplicates.txt"))))
        {
            for (Map.Entry<String, ArrayList<Path>> entry : collection.entrySet())
            {
                String key = entry.getKey();
                ArrayList<Path> value = entry.getValue();
                if (value.size() == 1) continue;

                bw.write(key);
                bw.newLine();
                for (Path path : value)
                {
                    bw.write("\t " + path.toString());
                    bw.newLine();
                }
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
