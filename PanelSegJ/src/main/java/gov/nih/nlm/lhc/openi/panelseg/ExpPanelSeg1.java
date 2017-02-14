package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;
import java.nio.file.Path;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * Created by jzou on 2/8/2017.
 */
public class ExpPanelSeg1 extends Exp
{
    public static void main(String args[])
    {
        log.info("Panel Segmentation with various methods.");

        ExpPanelSeg1 expPanelSeg = new ExpPanelSeg1();
        try
        {
            expPanelSeg.loadProperties();
            expPanelSeg.initialize();
            expPanelSeg.doWork();
            log.info("Completed!");
        }
        catch (Exception ex)
        {
            log.error(ex.getMessage());
        }
    }

    private PanelSeg.Method method;

    /**
     * Load the properties from ExpPanelSeg.properties file.
     * Also, validate all property values, throw exceptions if not valid.
     * @throws Exception
     */
    void loadProperties() throws Exception
    {
        loadProperties("ExpPanelSeg.properties");

        String propMethod = getProperty("Method");
        switch (propMethod)
        {
            case "LabelDetHog":
                method = PanelSeg.Method.LabelDetHog;
                loadPropertiesLabelDetHog();
                break;
            case "LabelRegHogSvm":
                method = PanelSeg.Method.LabelRegHogSvm;
                loadPropertiesLabelRegHogSvm();
                break;
//            case "LabelRegHogSvmThreshold":
//                method = PanelSeg.Method.LabelRegHogSvmThreshold;
//                break;
//            case "LabelRegHogSvmBeam":
//                method = PanelSeg.Method.LabelRegHogSvmBeam;
//                break;
//
            case "LabelDetHogLeNet5":
                method = PanelSeg.Method.LabelDetHogLeNet5;
                loadPropertiesLabelDetHogLeNet5();
                break;
//            case "LabelRegHogLeNet5Svm":
//                method = PanelSeg.Method.LabelRegHogLeNet5Svm;
//                break;
//            case "LabelRegHogLeNet5SvmBeam":
//                method = PanelSeg.Method.LabelRegHogLeNet5SvmBeam;
//                break;
//            case "LabelRegHogLeNet5SvmAlignment":
//                method = PanelSeg.Method.LabelRegHogLeNet5SvmAlignment;
//                break;
//
//            case "PanelSplitSantosh":
//                method = PanelSeg.Method.PanelSplitSantosh;
//                break;
//            case "PanelSplitJaylene":
//                method = PanelSeg.Method.PanelSplitJaylene;
//                break;
            default: throw new Exception("Method " + propMethod + " is Unknown");
        }
        waitKeyContinueOrQuit("Configuration Okay? Press any key to continue, press N to quit");
    }

    private void loadPropertiesLabelDetHog() throws Exception
    {
        propThreads = getProperty("Threads");
        propListFile = getProperty("ListFile");
        propTargetFolder = getProperty("TargetFolder");
        getProperty("LabelSetsHOG");
        getProperty("LabelHogModel");
    }

    private void loadPropertiesLabelRegHogSvm() throws Exception
    {
        propThreads = getProperty("Threads");
        propListFile = getProperty("ListFile");
        propTargetFolder = getProperty("TargetFolder");
        getProperty("LabelSetsHOG");
        getProperty("LabelHogModel");
        getProperty("LabelSvmModel");
    }

    private void loadPropertiesLabelDetHogLeNet5() throws Exception
    {
        propThreads = getProperty("Threads");
        propListFile = getProperty("ListFile");
        propTargetFolder = getProperty("TargetFolder");
        getProperty("LabelSetsHOG");
        getProperty("LabelHogModel");
        getProperty("LabelLeNet5Model");
    }

    @Override
    void initialize() throws Exception
    {
        setMultiThreading();
        setListFile();
        setTargetFolder(true);

        PanelSeg.initialize(method, properties);
    }

    @Override
    void doWork() throws Exception
    {
        if (threads > 1) doWorkMultiThread();
        else doWorkSingleThread();
    }

    @Override
    void doWork(int k) throws Exception
    {
        super.doWork(k);

        Path imagePath = imagePaths.get(k);
        //if (!imagePath.toString().endsWith("1471-2121-10-48-4.jpg")) return;

        opencv_core.Mat image = imread(imagePath.toString(), CV_LOAD_IMAGE_COLOR);
        List<Panel> panels = PanelSeg.segment(image, method);

        saveSegResult(imagePath.toFile().getName(), image, panels);
    }


}
