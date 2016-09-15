package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * HOG+SVM method for Label Recognition
 *
 * Created by jzou on 9/15/2016.
 */
class ExpLabelRecognitionHogSvm
{
    public static void main(String args[]) throws Exception {
        //Stop and print error msg if no arguments passed.
        if (args.length != 1) {
            System.out.println("Usage: java -cp PanelSegJ.jar ExpLabelRecognitionHogSvm <target folder>");
            System.out.println("	This is a utility program to Extract HoG features for HOG+SVM label recognition training.");
            System.exit(0);
        }

        ExpLabelRecognitionHogSvm generator = new ExpLabelRecognitionHogSvm(args[0]);
        generator.generate();
        System.out.println("Completed!");
    }

    private Path targetFolder;    //The folder for saving the result

    /**
     * Ctor, set targetFolder
     *
     * @param targetFolder
     */
    ExpLabelRecognitionHogSvm(String targetFolder) {
        this.targetFolder = Paths.get(targetFolder);
    }

    /**
     * Entry function
     */
    void generate()
    {
        List<Double> targets = new ArrayList<>();
        List<float[]> features = new ArrayList<>();

        PanelSegLabelRegHog hog = new PanelSegLabelRegHog();

        //Positive classes
        for (int i = 0; i < PanelSeg.labelChars.length; i++) {
            String name = PanelSeg.getLabelCharFolderName(PanelSeg.labelChars[i]);

            Path folder = targetFolder.resolve(name);
            List<Path> patches = AlgMiscEx.collectImageFiles(folder);

            for (Path path : patches)
            {
                opencv_core.Mat gray = imread(path.toString(), CV_LOAD_IMAGE_GRAYSCALE);
                float[] feature = hog.featureExtraction(gray);
                features.add(feature);
                targets.add((double)i);
            }
        }
        //Negative class
        {
            Path folder = targetFolder.resolve("neg");
            List<Path> patches = AlgMiscEx.collectImageFiles(folder);

            for (Path path : patches)
            {
                opencv_core.Mat gray = imread(path.toString(), CV_LOAD_IMAGE_GRAYSCALE);
                float[] feature = hog.featureExtraction(gray);
                features.add(feature);
                targets.add((double)PanelSeg.labelChars.length);
            }
        }

        Path folderModel = targetFolder.resolve("model");
        Path file = folderModel.resolve("train.txt");

        LibSvmEx.SaveInLibSVMFormat(file.toString(), targets, features);
    }

}
