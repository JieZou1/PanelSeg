package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;

import java.awt.*;
import java.nio.file.Path;
import java.util.ArrayList;

import static java.util.concurrent.ForkJoinTask.invokeAll;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 * Bootstrap of HOG+SVM method for Label Detection
 * Collect negative and positive patches for bootstrapping.
 *
 * Created by jzou on 9/9/2016.
 */
final class ExpLabelDetectHogBootstrap extends Exp
{
    public static void main(String args[]) throws Exception {
        //Stop and print error msg if no arguments passed.
        if (args.length != 2) {
            System.out.println("Usage: java -cp PanelSegJ.jar ExpLabelDetectHogBootstrap <Sample List File> <target folder>");
            System.out.println("	This is a utility program to Collect negative and positive patches for bootstrapping.");
            System.exit(0);
        }

        ExpLabelDetectHogBootstrap generator = new ExpLabelDetectHogBootstrap(args[0], args[1]);
        //generator.generateSingle();
        generator.generateMulti();
        System.out.println("Completed!");
    }

    /**
     * Ctor, set targetFolder and then collect all imagePaths
     * It also clean the targetFolder
     *
     * @param trainListFile
     * @param targetFolder
     */
    private ExpLabelDetectHogBootstrap(String trainListFile, String targetFolder) {
        super(trainListFile, targetFolder, false);

        for (String name : PanelSegLabelRegHog.labelSetsHOG)
        {
            Path folder = this.targetFolder.resolve(name);
            folder = folder.resolve("detected");
            AlgMiscEx.createClearFolder(folder);
        }
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
    }

    void generate(int k)
    {
        Path imagePath = imagePaths.get(k);
        String imageFile = imagePath.toString();
        System.out.println(Integer.toString(k) +  ": processing " + imageFile);

        PanelSegLabelRegHog hog = new PanelSegLabelRegHog();
        hog.segment(imageFile);

        //Save detected patches
        for (int i = 0; i < hog.hogDetectionResult.size(); i++)
        {
            ArrayList<Panel> segmentationResult = hog.hogDetectionResult.get(i);
            if (segmentationResult == null) continue;

            for (int j = 0; j < segmentationResult.size(); j++)
            {
                if (j == 3) break; //We just save the top 3 patches for training, in order to avoiding collecting a very large training set at the beginning.

                Panel panel = segmentationResult.get(j);
                Rectangle rectangle = panel.labelRect;

                opencv_core.Mat patch = AlgOpenCVEx.cropImage(hog.figure.imageGray, rectangle);
                panel.labelGrayNormPatch = new opencv_core.Mat();
                resize(patch, panel.labelGrayNormPatch, new opencv_core.Size(64, 64)); //Resize to 64x64 for easy browsing the results

                Path folder = targetFolder.resolve(panel.panelLabel).resolve("detected");
                Path file = folder.resolve(getLabelPatchFilename(imageFile, panel));
                opencv_imgcodecs.imwrite(file.toString(), panel.labelGrayNormPatch);
            }
        }
    }
}