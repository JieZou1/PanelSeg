package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;

import java.nio.file.Path;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * Created by jzou on 9/22/2016.
 */
public class ExpLabelRegHogSvmBeam extends Exp
{
    public static void main(String args[]) throws Exception {
        //Stop and print error msg if no arguments passed.
        if (args.length != 2) {
            System.out.println("Usage: java -cp PanelSegJ.jar ExpLabelRegHogSvmBeam <Sample List File> <target folder>");
            System.out.println("	This is a utility program to do Panel Label Recognition with HOG+SVM method.");
            System.out.println("	It saves recognition results (iPhotoDraw XML file) and preview images in target folder.");
            System.exit(0);
        }

        PanelSeg.initialize(PanelSeg.SegMethod.LabelRegHogSvmBeam);

        ExpLabelRegHogSvmBeam generator = new ExpLabelRegHogSvmBeam(args[0], args[1]);
        generator.segmentSingle();
        //generator.segmentMulti();
        System.out.println("Completed!");
    }

    /**
     * Ctor, set targetFolder and then collect all imagePaths
     * It also clean the targetFolder
     *
     * @param trainListFile
     * @param targetFolder
     */
    private ExpLabelRegHogSvmBeam(String trainListFile, String targetFolder) {
        super(trainListFile, targetFolder, true);
    }

    void generate(int k)
    {
        Path imagePath = imagePaths.get(k);
        System.out.println(Integer.toString(k) +  ": processing " + imagePath.toString());

        opencv_core.Mat image = imread(imagePath.toString(), CV_LOAD_IMAGE_COLOR);
        List<Panel> panels = PanelSeg.segment(image, PanelSeg.SegMethod.LabelRegHogSvmBeam);

        saveSegResult(imagePath.toFile().getName(), image, panels);
    }
}
