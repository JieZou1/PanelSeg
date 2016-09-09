package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_imgcodecs;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;

/**
 * Generate positive patches for training of HOG+SVM method for Label Detection
 * Negative patches are collected by Bootstrapping
 *
 * Created by jzou on 9/8/2016.
 */
final class ExpLabelDetectHogPos extends Exp {
    public static void main(String args[]) throws Exception {
        //Stop and print error msg if no arguments passed.
        if (args.length != 2) {
            System.out.println("Usage: java -cp PanelSegJ.jar ExpLabelDetectHogPos <Sample List File> <target folder>");
            System.out.println("	This is a utility program to generate positive training samples for label detection.");
            System.exit(0);
        }

        ExpLabelDetectHogPos generator = new ExpLabelDetectHogPos(args[0], args[1]);
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
    ExpLabelDetectHogPos(String trainListFile, String targetFolder) {
        super(trainListFile, targetFolder, false);
    }

    /**
     * Entry function for generating positive samples
     */
    void generate()
    {
        //Clean up all the folders
        Path posFolder = this.targetFolder;
        for (String name : PanelSegLabelRegHog.labelSetsHOG)
        {
            Path folder = posFolder.resolve(name);
            folder = folder.resolve("pos");
            AlgMiscEx.createClearFolder(folder);
        }

        for (int i = 0; i < imagePaths.size(); i++) {
            Path imagePath = imagePaths.get(i);
            String imageFile = imagePath.toString();
            String xmlFile = FilenameUtils.removeExtension(imageFile) + "_data.xml";

            //Load annotation
            File annotationFile = new File(xmlFile);
            ArrayList<Panel> panels = null;
            try {
                panels = iPhotoDraw.loadPanelSeg(annotationFile);
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }

            Figure figure = new Figure(imagePath, panels);
            figure.cropLabelGrayNormPatches(64, 64);

            for (Panel panel : figure.panels)
            {
                if (panel.labelRect == null) continue; //No label rect
                if (panel.panelLabel.length() != 1) continue; //Label has more than 1 char, we ignore for now.

                String name = PanelSegLabelRegHog.getLabelSetName(panel.panelLabel.charAt(0));
//                if (name == null)
//                {
//                    System.out.println(panel.panelLabel); continue;
//                }
                Path folder = posFolder.resolve(name).resolve("pos");

                String[] folderWords = imageFile.split("\\\\");
                name = folderWords[folderWords.length - 2] + "-" + folderWords[folderWords.length - 1] + "-";
                name += panel.labelRect.toString() + ".png";

                Path file = folder.resolve(name);

                opencv_imgcodecs.imwrite(file.toString(), panel.labelGrayNormPatch);
            }

        }
    }

}
