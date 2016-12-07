package gov.nih.nlm.lhc.openi.panelseg;

import ch.qos.logback.core.util.FileUtil;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import sun.net.www.protocol.file.FileURLConnection;

import java.io.*;
import java.nio.file.Path;
import java.util.*;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * Partition non graphic figures based on Jaylene's classification result
 *
 * Created by jzou on 12/7/2016.
 */
public class DataPartitionNonGraphics
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 0)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar PartitionNonGraphics");
            System.exit(0);
        }

        DataPartitionNonGraphics partitionNonGraphics = new DataPartitionNonGraphics();
        partitionNonGraphics.partition();
        System.out.println("Completed!");
    }

    private List<String> imagePaths;
    private HashMap<String, Integer> classificationResults;

    private void partition()
    {
        System.out.println("Read in Graphic Classification results");
        readGraphicClassificationResult();

        System.out.println("Read in eval.txt");
        readAllFigure("D:\\Users\\jie\\projects\\PanelSeg\\Exp\\eval.txt");

        System.out.println("Start Partitioning");

        List<String> graphics = new ArrayList<>();
        List<String> nonGraphics = new ArrayList<>();

        for (String path : imagePaths)
        {
            String file = FilenameUtils.getName(path);
            int classificationResult = classificationResults.get(file);
            if (classificationResult == 1)
                graphics.add(path);
            else if (classificationResult == 2)
                nonGraphics.add(path);
        }

        System.out.println("Save results");

        //save list
        String targetFolder = "D:\\Users\\jie\\projects\\PanelSeg\\Exp\\GraphicClassification\\";
        String graphicsFilename = targetFolder + "eval-graphics.txt";
        String nonGraphicsFilename = targetFolder + "eval-nongraphics.txt";
        writeList(graphicsFilename, graphics);
        writeList(nonGraphicsFilename, nonGraphics);

        //save images for preview
        targetFolder = "D:\\Users\\jie\\projects\\PanelSeg\\Exp\\GraphicClassification\\Graphics\\";
        for (String path : graphics)
        {
            opencv_core.Mat image = imread(path.toString(), CV_LOAD_IMAGE_COLOR);
            String file = FilenameUtils.getName(path);
            opencv_imgcodecs.imwrite(targetFolder + file, image);
        }
        targetFolder = "D:\\Users\\jie\\projects\\PanelSeg\\Exp\\GraphicClassification\\NonGraphics\\";
        for (String path : nonGraphics)
        {
            opencv_core.Mat image = imread(path.toString(), CV_LOAD_IMAGE_COLOR);
            String file = FilenameUtils.getName(path);
            opencv_imgcodecs.imwrite(targetFolder + file, image);
        }
    }

    private void readGraphicClassificationResult()
    {
        classificationResults = new HashMap<>();

        File folder = new File("\\Users\\jie\\projects\\PanelSeg\\Exp\\GraphicClassification");
        File[] listOfFiles = folder.listFiles();

        for (int i = 0; i < listOfFiles.length; i++)
        {
            if (listOfFiles[i].isFile())
            {
                readGraphicClassificationResult(listOfFiles[i]);
            }
        }
    }

    private void readGraphicClassificationResult(File file)
    {
        try (BufferedReader br = new BufferedReader(new FileReader(file)))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] words = line.split(",");
                String key = words[0];
                int value = Integer.parseInt(words[1]);
                classificationResults.put(key, value);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void readAllFigure(String listFile)
    {
        imagePaths = new ArrayList<String>();
        try {
            try (BufferedReader br = new BufferedReader(new FileReader(listFile))) {
                String line;
                while ((line = br.readLine()) != null)
                    imagePaths.add(line);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("Total number of image is: " + imagePaths.size());

    }

    private void writeList(String filename, List<String> list)
    {
        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename))))
        {
            for (String path : list)
            {
                bw.write(path);
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
