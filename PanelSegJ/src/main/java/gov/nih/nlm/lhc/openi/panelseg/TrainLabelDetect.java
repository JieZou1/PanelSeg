package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;

import java.awt.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 * For Label Detect Training
 * Refactored from ExpLabelDetect* classes
 *
 * Created by jzou on 11/9/2016.
 */
public class TrainLabelDetect extends Exp
{
    public enum Task {
        HogPos, HogSvmFeaExt, HogSvm2SingleVec, HogBootstrap
    }

    public static void main(String args[]) throws Exception {
        //Stop and print error msg if no arguments passed.
        if (args.length != 1) {
            System.out.println();

            System.out.println("Usage: java -cp PanelSegJ.jar TrainLabelDetect <Task>");
            System.out.println("Training tasks for Label Detection.");

            System.out.println();

            System.out.println("Task:");
            System.out.println("HogPos               HoG method for Label Detection");
            System.out.println("HogSvmFeaExt              HoG+SVM method for Label Recognition");
            System.out.println("HogSvm2SingleVec     HoG+SVM then followed by simple threshold for Label Recognition");
            System.out.println("HogBootstrap	        HoG+SVM then followed by beam search for Label Recognition");

            System.out.println();

            System.exit(0);
        }

        Task task = null;
        switch (args[0]) {
            case "HogPos":                task = Task.HogPos;                break;
            case "HogSvmFeaExt":          task = Task.HogSvmFeaExt;          break;
            case "HogSvm2SingleVec":      task = Task.HogSvm2SingleVec;      break;
            case "HogBootstrap":          task = Task.HogBootstrap;         break;
            default:
                System.out.println("Unknown method!!");
                System.exit(0);
        }

        String trainListFile, targetFolder;
        trainListFile = "\\Users\\jie\\projects\\PanelSeg\\Exp\\train.txt";
        targetFolder = "\\Users\\jie\\projects\\PanelSeg\\Exp\\LabelDetect\\Hog";

        TrainLabelDetect train = new TrainLabelDetect(trainListFile, targetFolder, task);

        switch (task)
        {
            case HogPos:
            case HogBootstrap:
                //train.doWorkSingleThread();
                train.doWorkMultiThread();
                break;
            case HogSvmFeaExt:
                train.doWorkHogSvmFeaExt();
                break;
            case HogSvm2SingleVec:
                train.doWorkHogSvm2SingleVec();
        }
        System.out.println("Completed!");
    }

    private Task task;

    TrainLabelDetect(String trainListFile, String targetFolder, Task task)
    {
        super(trainListFile, targetFolder, false);
        this.task = task;
        switch (task)
        {
            case HogPos: initializeHogPos(); break;
            case HogSvmFeaExt: break;
            case HogSvm2SingleVec: break;
            case HogBootstrap: initializeHogBootstrap(); break;
        }
    }

    void doWork(int i)
    {
        switch (task)
        {
            case HogPos: doWorkHogPos(i); break;
            case HogBootstrap: doWorkHogBootstrap(i); break;
        }
    }

    //region Crop positive patches from training figures.
    private void initializeHogPos()
    {
        //Clean up all the folders
        for (String name : LabelDetectHog.labelSetsHOG) {
            Path folder = this.targetFolder.resolve(name);
            folder = folder.resolve("pos");
            AlgMiscEx.createClearFolder(folder);
        }
    }
    private void doWorkHogPos(int i)
    {
        Path imagePath = imagePaths.get(i);
        String imageFile = imagePath.toString();
        //System.out.println(Integer.toString(i) +  ": processing " + imageFile);

        //Load annotation
        String xmlFile = FilenameUtils.removeExtension(imageFile) + "_data.xml";
        File annotationFile = new File(xmlFile);
        ArrayList<Panel> panels = null;
        try {
            panels = iPhotoDraw.loadPanelSegGt(annotationFile);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

        Figure figure = new Figure(imagePath, panels);
        figure.cropLabelGrayNormPatches(64, 64);

        for (Panel panel : figure.panels)
        {
            if (panel.labelRect == null) continue; //No label rect
            if (panel.panelLabel.length() != 1) continue; //Label has more than 1 char, we ignore for now.

            String name = LabelDetectHog.getLabelSetName(panel.panelLabel.charAt(0));
//                if (name == null)
//                {
//                    System.out.println(panel.panelLabel); continue;
//                }
            Path folder = targetFolder.resolve(name).resolve("pos");
            Path file = folder.resolve(getLabelPatchFilename(imageFile, panel));
            opencv_imgcodecs.imwrite(file.toString(), panel.labelGrayNormPatch);
        }
    }
    //endregion

    //region Hog SVM feature extraction, and generate libsvm training file.

    private void doWorkHogSvmFeaExt()
    {
        for (int i = 0; i <  LabelDetectHog.labelSetsHOG.length; i++) {
            String name = LabelDetectHog.labelSetsHOG[i];

            Path folder = targetFolder.resolve(name);
            Path folderPos = folder.resolve("pos");
            Path folderNeg = folder.resolve("neg");
            Path folderModel = folder.resolve("model");

            List<Path> posPatches = AlgMiscEx.collectImageFiles(folderPos);
            List<Path> negPatches = AlgMiscEx.collectImageFiles(folderNeg);

            Path file = folderModel.resolve("train.txt");
            double[] targets = new double[posPatches.size() + negPatches.size()];
            float[][] features = new float[posPatches.size() + negPatches.size()][];

            int k = 0;
            for (Path path : posPatches) {
                opencv_core.Mat gray = imread(path.toString(), CV_LOAD_IMAGE_GRAYSCALE);
                float[] feature = LabelDetectHog.featureExtraction(gray);
                features[k] = feature;
                targets[k] = 1.0;
                k++;
            }
            for (Path path : negPatches) {
                opencv_core.Mat gray = imread(path.toString(), CV_LOAD_IMAGE_GRAYSCALE);
                float[] feature = LabelDetectHog.featureExtraction(gray);
                features[k] = feature;
                targets[k] = 0.0;
                k++;
            }

            LibSvmEx.SaveInLibSVMFormat(file.toString(), targets, features);
        }
    }
    //endregion

    //region Hog Bootstrapping

    private void initializeHogBootstrap()
    {
        for (String name : LabelDetectHog.labelSetsHOG)
        {
            Path folder = this.targetFolder.resolve(name);
            folder = folder.resolve("detected");
            AlgMiscEx.createClearFolder(folder);
        }
    }

    private void doWorkHogBootstrap(int k)
    {
        Path imagePath = imagePaths.get(k);
        String imageFile = imagePath.toString();
        System.out.println(Integer.toString(k) +  ": processing " + imageFile);

        Figure figure = new Figure(imageFile);
        LabelDetectHog hog = new LabelDetectHog(figure);
        hog.hoGDetect();

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
    //endregion

    //region convert Hog SVM model to single vector format

    private void doWorkHogSvm2SingleVec()
    {
        Path vectorPath = targetFolder.resolve("vector.java");

        //Save to a java file
        try (PrintWriter pw = new PrintWriter(vectorPath.toString()))
        {
            pw.println("package gov.nih.nlm.lhc.openi.panelseg;");
            pw.println();

            for (String name : LabelDetectHog.labelSetsHOG)
            {
                Path folder = targetFolder.resolve(name);
                Path folderModel = folder.resolve("model");
                Path modelPath = folderModel.resolve("svm_model");

                float[] singleVector = LibSvmEx.ToSingleVector(modelPath.toString());

                String classname =	"PanelSegLabelRegHoGModel_" + name;

                pw.println("final class " + classname);
                pw.println("{");

                pw.println("	static float[] svmModel = ");

                pw.println("    	{");
                for (int k = 0; k < singleVector.length; k++)
                {
                    pw.print(singleVector[k] + "f,");
                }
                pw.println();
                pw.println("    };");
                pw.println("}");
            }
        }
        catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
    //endregion
}
