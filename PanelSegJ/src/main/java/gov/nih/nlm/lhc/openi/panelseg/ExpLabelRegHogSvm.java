package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;

import java.nio.file.Path;
import java.util.List;

import static java.util.concurrent.ForkJoinTask.invokeAll;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * Experiments of HOG+SVM method for Label Detection
 *
 * Created by jzou on 9/15/2016.
 */
public class ExpLabelRegHogSvm extends  Exp{
    public static void main(String args[]) throws Exception {
        //Stop and print error msg if no arguments passed.
        if (args.length != 2) {
            System.out.println("Usage: java -cp PanelSegJ.jar ExpLabelRegHogSvm <Sample List File> <target folder>");
            System.out.println("	This is a utility program to do Panel Label Recognition with HOG+SVM method.");
            System.out.println("	It saves recognition results (iPhotoDraw XML file) and preview images in target folder.");
            System.exit(0);
        }

        ExpLabelRegHogSvm generator = new ExpLabelRegHogSvm(args[0], args[1]);

        PanelSeg.initialze(PanelSeg.SegMethed.LabelRegHogSvm);

        generator.generateSingle();
        //generator.generateMulti();
        System.out.println("Completed!");
    }

    /**
     * Ctor, set targetFolder and then collect all imagePaths
     * It also clean the targetFolder
     *
     * @param trainListFile
     * @param targetFolder
     */
    private ExpLabelRegHogSvm(String trainListFile, String targetFolder) {
        super(trainListFile, targetFolder, true);
    }

    /**
     * Entry function
     */
    void generateSingle()
    {
        for (int k = 0; k < imagePaths.size(); k++) generate(k);
    }

    private void generateMulti()
    {
        ExpTask[] tasks = ExpTask.createTasks(this, imagePaths.size(), 4);
        invokeAll(tasks);
//        ExpTask task = new ExpTask(this, 0, imagePaths.size());
//        task.invoke();
    }

    void generate(int k)
    {
        Path imagePath = imagePaths.get(k);
        System.out.println(Integer.toString(k) +  ": processing " + imagePath.toString());

        opencv_core.Mat image = imread(imagePath.toString(), CV_LOAD_IMAGE_COLOR);
        List<Panel> panels = PanelSeg.segment(image, PanelSeg.SegMethed.LabelRegHogSvm);

        //Save result in iPhotoDraw XML file
        String xmlFile = FilenameUtils.removeExtension(imagePath.toFile().getName()) + "_data.xml";
        Path xmlPath = targetFolder.resolve(xmlFile);
        try {
            iPhotoDraw.savePanelSeg(xmlPath.toFile(), panels);
        } catch (Exception e) {
            e.printStackTrace();
        }

        //Save preview in jpg file
        Path jpgPath = targetFolder.resolve(imagePath.toFile().getName());
        opencv_core.Mat jpg = PanelSeg.drawAnnotation(image, panels);
        opencv_imgcodecs.imwrite(jpgPath.toString(), jpg);
    }

}
