package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;

import java.awt.*;
import java.io.File;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 * For HoG Label Detect Training
 *
 * Created by jzou on 2/10/2017.
 */
public class ExpTrainLabelDetectHog extends Exp
{
    public static void main(String args[])
    {
        log.info("Training tasks for HOG-based Label Detection.");

        ExpTrainLabelDetectHog exp = new ExpTrainLabelDetectHog();
        try
        {
            exp.loadProperties();
            exp.initialize();
            exp.doWork();
            log.info("Completed!");
        }
        catch (Exception ex)
        {
            log.error(ex.getMessage());
        }
    }

    public enum Task { HogPos, HogSvmFeaExt, HogSvm2SingleVec, HogBootstrap }
    private ExpTrainLabelDetectHog.Task task;

    private String propLabelSetsHOG;
    private String propPosGtFolder, propPosFolder, propNegFolder, propModelFolder, propDetectedFolder;
    private String propTrainFile, propSvmModelFile, propVectorSvmFile;

    private void setPosGtFolder(boolean toCleanFolder) throws Exception
    {
        if (toCleanFolder)
        {
            for (String name : LabelDetectHog.labelSetsHOG)
            {
                Path folder = this.targetFolder.resolve(name);
                folder = folder.resolve(propPosGtFolder);
                System.out.println(folder + " is going to be cleaned!");
            }
            waitKeyContinueOrQuit("Press any key to continue, press N to quit");

            //Clean up all the folders
            for (String name : LabelDetectHog.labelSetsHOG) {
                Path folder = this.targetFolder.resolve(name);
                folder = folder.resolve(propPosGtFolder);
                AlgMiscEx.createClearFolder(folder);
                log.info("Folder " + folder + " is created or cleaned!");
            }
        }
    }

    private void setDetectFolder(boolean toCleanFolder) throws Exception
    {
        if (toCleanFolder)
        {
            for (String name : LabelDetectHog.labelSetsHOG)
            {
                Path folder = this.targetFolder.resolve(name);
                folder = folder.resolve(propDetectedFolder);
                System.out.println(folder + " is going to be cleaned!");
            }
            waitKeyContinueOrQuit("Press any key to continue, press N to quit");

            //Clean up all the folders
            for (String name : LabelDetectHog.labelSetsHOG) {
                Path folder = this.targetFolder.resolve(name);
                folder = folder.resolve(propDetectedFolder);
                AlgMiscEx.createClearFolder(folder);
                log.info("Folder " + folder + " is created or cleaned!");
            }
        }
    }

    private void setLabelSetsHOG() throws Exception
    {
        switch (propLabelSetsHOG) {
            case "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz123456789":
                LabelDetectHog.labelSetsHOG = new String[1];
                LabelDetectHog.labelSetsHOG[0] = propLabelSetsHOG;
                break;
            default:
                throw new Exception("labelSetsHOG " + propLabelSetsHOG + " is Unknown");
        }
    }

    @Override
    void loadProperties() throws Exception
    {
        loadProperties("ExpTrainLabelDetectHog.properties");

        String strTask = setProperty("Task");
        switch (strTask)
        {
            case "HogPos":
                task = ExpTrainLabelDetectHog.Task.HogPos;
                log.info("Collect Positive Patches from Ground-Truth Annotation.");
                log.info("It reads ListFile, crop and normalize images, and then save the result to PosGtFolder.");
                loadPropertiesHogPos();
                break;
            case "HogSvmFeaExt":
                task = ExpTrainLabelDetectHog.Task.HogSvmFeaExt;
                log.info("SVM feature extraction.");
                log.info("It reads from PosFolder, NegFolder and then save libsvm training file to ModelFolder.");
                loadPropertiesHogSvmFeaExt();
                break;
            case "HogSvm2SingleVec":
                task = ExpTrainLabelDetectHog.Task.HogSvm2SingleVec;
                log.info("Convert Linear SVM model to Single-Vector format.");
                log.info("It converts Linear SVM model file SvmModelFile and save to VectorSvmFile.");
                loadPropertiesHogSvm2SingleVec();
                break;
            case "HogBootstrap":
                task = ExpTrainLabelDetectHog.Task.HogBootstrap;
                log.info("Bootstrapping");
                log.info("It reads ListFile, does HOG detection, and save results to DetectedFolder.");
                loadPropertiesHogBootstrap();
                break;
            default: throw new Exception("Task " + strTask + " is Unknown");
        }

        waitKeyContinueOrQuit("Configuration Okay? Press any key to continue, press N to quit");
    }

    @Override
    void initialize() throws Exception
    {
        switch (task)
        {
            case HogPos: initializeHogPos(); break;
            case HogSvmFeaExt: initializeHogSvmFeaExt(); break;
            case HogSvm2SingleVec: initializeHogSvm2SingleVec(); break;
            case HogBootstrap: initializeHogBootstrap(); break;
        }
    }

    @Override
    void doWork() throws Exception
    {
        super.doWork();

        //Do work
        switch (task)
        {
            case HogPos:
            case HogBootstrap:
                if (threads > 1) doWorkMultiThread();
                else doWorkSingleThread();
                break;
            case HogSvmFeaExt:                doWorkHogSvmFeaExt();     break;
            case HogSvm2SingleVec:            doWorkHogSvm2SingleVec(); break;
        }
    }

    @Override
    void doWork(int k) throws Exception
    {
        super.doWork(k);
        switch (task)
        {
            case HogPos: doWorkHogPos(k); break;
            case HogBootstrap: doWorkHogBootstrap(k); break;
        }
    }

    private void loadPropertiesHogPos() throws Exception
    {
        propThreads = setProperty("Threads");
        propListFile = setProperty("ListFile");
        propLabelSetsHOG = setProperty("LabelSetsHOG");
        propTargetFolder = setProperty("TargetFolder");
        propPosGtFolder = setProperty("PosGtFolder");
    }

    private void initializeHogPos() throws Exception
    {
        setMultiThreading();
        setListFile();
        setLabelSetsHOG();
        setTargetFolder();
        setPosGtFolder(true);
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
            Path folder = targetFolder.resolve(name).resolve(propPosGtFolder);
            Path file = folder.resolve(getLabelPatchFilename(imageFile, panel));
            opencv_imgcodecs.imwrite(file.toString(), panel.labelGrayNormPatch);
        }
    }

    private void loadPropertiesHogSvmFeaExt() throws Exception
    {
        propTargetFolder = setProperty("TargetFolder");
        propLabelSetsHOG = setProperty("LabelSetsHOG");
        propModelFolder = setProperty("ModelFolder");
        propPosFolder = setProperty("PosFolder");
        propNegFolder = setProperty("NegFolder");
        propTrainFile = setProperty("TrainFile");
    }

    private void initializeHogSvmFeaExt() throws Exception
    {
        setLabelSetsHOG();
        setTargetFolder();
    }

    private void doWorkHogSvmFeaExt()
    {
        for (int i = 0; i <  LabelDetectHog.labelSetsHOG.length; i++) {
            String name = LabelDetectHog.labelSetsHOG[i];

            Path folder = targetFolder.resolve(name);
            Path folderPos = folder.resolve(propPosFolder);
            Path folderNeg = folder.resolve(propNegFolder);
            Path folderModel = folder.resolve(propModelFolder);

            List<Path> posPatches = AlgMiscEx.collectImageFiles(folderPos);
            List<Path> negPatches = AlgMiscEx.collectImageFiles(folderNeg);

            Path file = folderModel.resolve(propTrainFile);
            double[] targets = new double[posPatches.size() + negPatches.size()];
            float[][] features = new float[posPatches.size() + negPatches.size()][];

            int k = 0;
            log.info("Computing Features for Positive Samples.");
            for (Path path : posPatches) {
                opencv_core.Mat gray = imread(path.toString(), CV_LOAD_IMAGE_GRAYSCALE);
                float[] feature = LabelDetectHog.featureExtraction(gray);
                features[k] = feature;
                targets[k] = 1.0;
                k++;
            }
            log.info("Totally, " + k + " Positive Samples.");
            log.info("Computing Features for Negative Samples.");
            for (Path path : negPatches) {
                opencv_core.Mat gray = imread(path.toString(), CV_LOAD_IMAGE_GRAYSCALE);
                float[] feature = LabelDetectHog.featureExtraction(gray);
                features[k] = feature;
                targets[k] = 0.0;
                k++;
            }
            log.info("Totally, " + k + " Positive and Negative Samples.");

            log.info("Saving into LibSVM format to " + file.toString());
            LibSvmEx.SaveInLibSVMFormat(file.toString(), targets, features);
        }
    }

    private void loadPropertiesHogSvm2SingleVec() throws Exception
    {
        propTargetFolder = setProperty("TargetFolder");
        propLabelSetsHOG = setProperty("LabelSetsHOG");
        propModelFolder = setProperty("ModelFolder");
        propSvmModelFile = setProperty("SvmModelFile");
        propVectorSvmFile = setProperty("VectorSvmFile");
    }

    private void initializeHogSvm2SingleVec() throws Exception
    {
        setLabelSetsHOG();
        setTargetFolder();
    }

    private void doWorkHogSvm2SingleVec() throws  Exception
    {
        Path vectorPath = targetFolder.resolve(propVectorSvmFile);
        log.info("Save Single Vector SVM to " + vectorPath.toString());

        //Save to a java file
        try (PrintWriter pw = new PrintWriter(vectorPath.toString()))
        {
            pw.println("package gov.nih.nlm.lhc.openi.panelseg;");
            pw.println();

            for (String name : LabelDetectHog.labelSetsHOG)
            {
                Path folder = targetFolder.resolve(name);
                Path folderModel = folder.resolve(propModelFolder);
                Path modelPath = folderModel.resolve(propSvmModelFile);

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
    }

    private void loadPropertiesHogBootstrap() throws Exception
    {
        propThreads = setProperty("Threads");
        propListFile = setProperty("ListFile");
        propTargetFolder = setProperty("TargetFolder");
        propLabelSetsHOG = setProperty("LabelSetsHOG");
        propDetectedFolder = setProperty("DetectedFolder");
    }

    private void initializeHogBootstrap() throws Exception
    {
        setMultiThreading();
        setListFile();
        setTargetFolder();
        setDetectFolder(true);
    }

    private void doWorkHogBootstrap(int k)
    {
        Path imagePath = imagePaths.get(k);
        String imageFile = imagePath.toString();

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

                Path folder = targetFolder.resolve(panel.panelLabel).resolve(propDetectedFolder);
                Path file = folder.resolve(getLabelPatchFilename(imageFile, panel));
                opencv_imgcodecs.imwrite(file.toString(), panel.labelGrayNormPatch);
            }
        }
    }

}
