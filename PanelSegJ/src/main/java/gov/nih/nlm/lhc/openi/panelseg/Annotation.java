package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * The base class for all annotation related operations on one data set. <p>
 * This class is not intended to be instantiated, so we make it abstract.
 *
 * Created by jzou on 8/26/2016.
 */
public abstract class Annotation
{
    protected Path annotationFolder;        //Annotation folder of a set
    protected ArrayList<Path> imagePaths;   //The image file path of the set
    protected Path stylePath;               //The style file path
    protected Map<String, String> styles;   //The styles of the figures

    /**
     * Ctor, set annotationFolder and then collect all imagePaths
     * It also load the style annotation into styles.
     * If style.txt is not found in the annotationFolder, styles is set to null.
     * @param annotationFolder
     */
    protected Annotation(String annotationFolder)
    {
        this.annotationFolder = Paths.get(annotationFolder);
        imagePaths = AlgorithmEx.CollectImageFiles(this.annotationFolder);
        System.out.println("Total number of image is: " + imagePaths.size());

        stylePath = Paths.get(annotationFolder, "style.txt");
        styles = Annotation.loadStyleMap(stylePath);
    }

    /**
     * Draw annotation onto the image for viewing and saving purpose
     * @param img
     * @param panels
     * @return
     */
    static protected opencv_core.Mat drawAnnotation(opencv_core.Mat img, ArrayList<Panel> panels, String style)
    {
        opencv_core.Mat imgAnnotated = new opencv_core.Mat();
        opencv_core.copyMakeBorder(img, imgAnnotated, 0, 100, 0, 50, opencv_core.BORDER_CONSTANT, new opencv_core.Scalar());

        //Draw bounding box first
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            opencv_core.Scalar scalar = AlgorithmEx.getColor(i);

            opencv_core.Rect panel_rect = AlgorithmEx.Rectangle2Rect(panel.panelRect);
            opencv_imgproc.rectangle(imgAnnotated, panel_rect, scalar, 3, 8, 0);

            if (panel.panelLabel.length() != 0)
            {
                opencv_core.Rect label_rect = AlgorithmEx.Rectangle2Rect(panel.labelRect);
                opencv_imgproc.rectangle(imgAnnotated, label_rect, scalar, 1, 8, 0);
            }
        }

        //Draw labels to make the text stand out.
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            opencv_core.Scalar scalar = AlgorithmEx.getColor(i);

            if (panel.panelLabel.length() != 0)
            {
                opencv_core.Rect label_rect = AlgorithmEx.Rectangle2Rect(panel.labelRect);
                opencv_core.Point bottom_left = new opencv_core.Point(label_rect.x() + label_rect.width(), label_rect.y() + label_rect.height() + 50);
                opencv_imgproc.putText(imgAnnotated, panel.panelLabel, bottom_left, opencv_imgproc.CV_FONT_HERSHEY_PLAIN, 5, scalar, 3, 8, false);
            }
        }

        {//Draw Style Annotation
            opencv_core.Scalar scalar = AlgorithmEx.getColor(1);
            opencv_core.Point bottom_left = new opencv_core.Point(0, img.rows() + 100);
            opencv_imgproc.putText(imgAnnotated, style, bottom_left, opencv_imgproc.CV_FONT_HERSHEY_PLAIN, 2, scalar, 3, 8, false);
        }

        return imgAnnotated;
    }

    /**
     * Read the style annotation from the file.
     * If the styleFile does not exist, return null.
     *
     * @param stylePath The filepath
     * @return image file name string and style annotation in HashMap<string, string>
     */
    static HashMap<String, String> loadStyleMap(Path stylePath)
    {
        HashMap<String, String> styles = new HashMap<>();

        if(!Files.exists(stylePath) || Files.isDirectory(stylePath))
        {	//No styles have been marked yet
            System.out.println("Not able to find style.txt!");
            return null;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(stylePath.toFile())))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] words = line.split("\\s+");
                styles.put(words[0], words[1]);
            }
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return styles;
    }

    /**
     * Save the style annotation into the file
     * @param stylePath
     * @param styles
     */
    static void saveStyleMap(Path stylePath, Map<String, String> styles)
    {
        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(stylePath.toFile()))))
        {
            for (Map.Entry<String, String> entry : styles.entrySet())
            {
                String key = entry.getKey();
                String value = entry.getValue();
                bw.write(key + "\t " + value);
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
