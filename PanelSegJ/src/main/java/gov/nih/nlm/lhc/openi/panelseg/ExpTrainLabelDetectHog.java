package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 * Refactor from TrainLabelDetect to use Properties to configure the training.
 *
 * Created by jzou on 2/8/2017.
 */
public class ExpTrainLabelDetectHog extends Exp
{
    protected static final Logger log = LoggerFactory.getLogger(ExpPanelSeg1.class);

    public enum Task { HogPos, HogSvmFeaExt, HogSvm2SingleVec, HogBootstrap }

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

    private ExpTrainLabelDetectHog.Task task;

    private ExpTrainLabelDetectHog() {super();}

    /**
     * Load the properties from ExpPanelSeg.properties file.
     * Also, validate all property values, throw exceptions if not valid.
     * @throws Exception
     */
    private void loadProperties() throws Exception
    {
        //Load properties
        properties = new Properties();
        properties.load(this.getClass().getClassLoader().getResourceAsStream("ExpTrainLabelDetectHog.properties"));

        String strListFile = properties.getProperty("listFile");
        if (strListFile == null) throw new Exception("ERROR: listFile property is Missing.");
        File list_file = new File(strListFile);
        if (!list_file.exists()) throw new Exception("ERROR: " + strListFile + " does not exist.");
        if (!list_file.isFile()) throw new Exception("ERROR: " + strListFile + " is not a file.");

        log.info("listFile: " + strListFile);
        listFile = list_file.toPath();
        loadListFile();
        log.info("Total number of image is: " + imagePaths.size());

        String strTargetFolder = properties.getProperty("targetFolder");
        if (strTargetFolder == null) throw new Exception("ERROR: targetFolder property is Missing.");
        File target_folder = new File(strTargetFolder);
        log.info("targetFolder: " + strTargetFolder);
        targetFolder = target_folder.toPath();

        String strTask = properties.getProperty("task");
        if (strTask == null) throw new Exception("ERROR: task property is Missing.");
        switch (strTask)
        {
            case "HogPos":                task = Task.HogPos;                break;
            case "HogSvmFeaExt":          task = Task.HogSvmFeaExt;          break;
            case "HogSvm2SingleVec":      task = Task.HogSvm2SingleVec;      break;
            case "HogBootstrap":          task = Task.HogBootstrap;          break;
            default: throw new Exception(strTask + " is Unknown");
        }
        log.info("Task: " + strTask);

        String strlabelSetsHOG = properties.getProperty("labelSetsHOG");
        if (strlabelSetsHOG == null) throw new Exception("ERROR: labelSetsHOG property is Missing.");
        log.info("labelSetsHOG: " + strlabelSetsHOG);
        switch (strlabelSetsHOG)
        {
            case "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz123456789":
                LabelDetectHog.labelSetsHOG = new String[1];
                LabelDetectHog.labelSetsHOG[0] = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz123456789";
        }
    }

    @Override
    void initialize() throws Exception {
        switch (task)
        {
            case HogPos: initializeHogPos(); break;
            case HogSvmFeaExt: break;
            case HogSvm2SingleVec: break;
            case HogBootstrap: initializeHogBootstrap(); break;
        }
    }

    private void initializeHogPos()
    {
        //Clean up all the folders
        for (String name : LabelDetectHog.labelSetsHOG) {
            Path folder = this.targetFolder.resolve(name);
            folder = folder.resolve("pos");
            AlgMiscEx.createClearFolder(folder);
        }
    }

    private void initializeHogBootstrap()
    {
        for (String name : LabelDetectHog.labelSetsHOG)
        {
            Path folder = this.targetFolder.resolve(name);
            folder = folder.resolve("detected");
            AlgMiscEx.createClearFolder(folder);
        }
    }

    @Override
    void doWork() throws Exception {
        switch (task)
        {
            case HogPos:
            case HogBootstrap:
                //train.doWorkSingleThread();
                doWorkMultiThread();
                break;
            case HogSvmFeaExt:
                doWorkHogSvmFeaExt();
                break;
            case HogSvm2SingleVec:
                doWorkHogSvm2SingleVec();
        }
    }

    @Override
    void doWork(int k) throws Exception {
        switch (task)
        {
            case HogPos: doWorkHogPos(k); break;
            case HogBootstrap: doWorkHogBootstrap(k); break;
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
}
