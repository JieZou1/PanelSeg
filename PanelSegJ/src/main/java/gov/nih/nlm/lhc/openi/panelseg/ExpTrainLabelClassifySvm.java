package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * Created by jzou on 2/14/2017.
 */
public class ExpTrainLabelClassifySvm extends Exp
{
    public static void main(String args[])
    {
        log.info("Training tasks for SVM Label Classification.");

        ExpTrainLabelClassifySvm exp = new ExpTrainLabelClassifySvm();
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

    private String propNumberOfClasses;

    @Override
    void loadProperties() throws Exception
    {
        loadProperties("ExpTrainLabelClassifySvm.properties");

        propTrainFolder = getProperty("TrainFolder");
        propTargetFile = getProperty("TargetFile");
        propNumberOfClasses = getProperty("NumberOfClasses");

        waitKeyContinueOrQuit("Configuration Okay? Press any key to continue, press N to quit");
    }

    @Override
    void initialize() throws Exception {
        setTrainFolder();
    }

    @Override
    void doWork() throws Exception
    {
        doWorkHogSvmFeaExt();
    }

    private void doWorkHogSvmFeaExt()
    {
        int numberOfClasses = Integer.parseInt(propNumberOfClasses);

        List<Double> targets = new ArrayList<>();
        List<float[]> features = new ArrayList<>();

        //Positive classes
        for (int i = 0; i < PanelSeg.labelChars.length; i++) {
            String name = PanelSeg.getLabelCharFolderName(PanelSeg.labelChars[i]);

            Path folder = trainFolder.resolve(name);
            List<Path> patches = AlgMiscEx.collectImageFiles(folder);

            for (Path path : patches)
            {
                opencv_core.Mat gray = imread(path.toString(), CV_LOAD_IMAGE_GRAYSCALE);
                float[] feature =  LabelDetectHog.featureExtraction(gray);
                features.add(feature);
                targets.add((double)i);
            }
        }

        if (numberOfClasses == 51)
        {   //Negative class
            Path folder = trainFolder.resolve("neg");
            List<Path> patches = AlgMiscEx.collectImageFiles(folder);

            for (Path path : patches)
            {
                opencv_core.Mat gray = imread(path.toString(), CV_LOAD_IMAGE_GRAYSCALE);
                float[] feature = LabelDetectHog.featureExtraction(gray);
                features.add(feature);
                targets.add((double)PanelSeg.labelChars.length);
            }
        }

        LibSvmEx.SaveInLibSVMFormat(propTargetFile, targets, features);
    }
}
