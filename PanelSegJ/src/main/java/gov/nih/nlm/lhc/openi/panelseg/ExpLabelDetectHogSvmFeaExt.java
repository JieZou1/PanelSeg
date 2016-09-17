package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;

import java.nio.file.Path;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * HOG+SVM method for Label Detection
 *
 * Created by jzou on 9/9/2016.
 */
final class ExpLabelDetectHogSvmFeaExt extends Exp
{
    public static void main(String args[]) throws Exception {
        //Stop and print error msg if no arguments passed.
        if (args.length != 2) {
            System.out.println("Usage: java -cp PanelSegJ.jar ExpLabelDetectHogSvmFeaExt <Sample List File> <target folder>");
            System.out.println("	This is a utility program to Extract HoG features for HOG+SVM label detection training.");
            System.exit(0);
        }

        ExpLabelDetectHogSvmFeaExt generator = new ExpLabelDetectHogSvmFeaExt(args[0], args[1]);
        generator.generate();
        System.out.println("Completed!");
    }

    /**
     * Ctor, set targetFolder and then collect all imagePaths
     * It also clean the targetFolder
     *
     * @param trainListFile
     * @param targetFolder
     */
    ExpLabelDetectHogSvmFeaExt(String trainListFile, String targetFolder) {
        super(trainListFile, targetFolder, false);
    }

    /**
     * Entry function
     */
    void generate()
    {
        for (int i = 0; i < PanelSegLabelRegHog.labelSetsHOG.length; i++) generate(i);
    }

    void generate(int i)
    {
        String name = PanelSegLabelRegHog.labelSetsHOG[i];

        Path folder = targetFolder.resolve(name);
        Path folderPos = folder.resolve("pos");
        Path folderNeg = folder.resolve("neg");
        Path folderModel = folder.resolve("model");

        List<Path> posPatches = AlgMiscEx.collectImageFiles(folderPos);
        List<Path> negPatches = AlgMiscEx.collectImageFiles(folderNeg);

        Path file = folderModel.resolve("train.txt");
        double[] targets = new double[posPatches.size() + negPatches.size()];
        float[][] features = new float[posPatches.size() + negPatches.size()][];

        PanelSegLabelRegHog hog = new PanelSegLabelRegHog();

        int k = 0;
        for (Path path : posPatches)
        {
            opencv_core.Mat gray = imread(path.toString(), CV_LOAD_IMAGE_GRAYSCALE);
            float[] feature = hog.featureExtraction(gray);
            features[k] = feature;
            targets[k] = 1.0;
            k++;
        }
        for (Path path : negPatches)
        {
            opencv_core.Mat gray = imread(path.toString(), CV_LOAD_IMAGE_GRAYSCALE);
            float[] feature = hog.featureExtraction(gray);
            features[k] = feature;
            targets[k] = 0.0;
            k++;
        }

        LibSvmEx.SaveInLibSVMFormat(file.toString(), targets, features);
    }

}
