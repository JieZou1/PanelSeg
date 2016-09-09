package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_imgcodecs;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;

/**
 * Created by jzou on 8/30/2016.
 */
final class ExpLabelPreview extends Exp {
    public static void main(String args[]) throws Exception {
        //Stop and print error msg if no arguments passed.
        if (args.length != 2) {
            System.out.println("Usage: java -cp PanelSegJ.jar ExpLabelDetectHogPos <Sample List File> <target folder>");
            System.out.println("	This is a utility program to generate preview of label patches.");
            System.out.println("	It can also be used for generate positive training samples for label detection.");
            System.exit(0);
        }

        ExpLabelPreview preview = new ExpLabelPreview(args[0], args[1]);
        preview.generatePreview();
        System.out.println("Completed!");
    }

    /**
     * Ctor, set targetFolder and then collect all imagePaths
     * It also clean the targetFolder
     *
     * @param trainListFile
     * @param targetFolder
     */
    ExpLabelPreview(String trainListFile, String targetFolder) {
        super(trainListFile, targetFolder, true);
    }

    /**
     * Entry function for generating label preview
     */
    void generatePreview()
    {
        generatePreview(LabelPreviewType.ORIGINAL);
        generatePreview(LabelPreviewType.NORM64);
    }

    /**
     * Generate original label preview
     */
    void generatePreview(LabelPreviewType type)
    {
        //Clean up all the folders
        Path typeFolder = this.targetFolder.resolve(type.toString());
        AlgMiscEx.createClearFolder(typeFolder);
        for (char c : PanelSeg.labelChars)
        {
            String name = PanelSeg.getLabelCharFolderName(c);
            Path folder = typeFolder.resolve(name);
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
            switch (type)
            {
                case ORIGINAL: figure.cropLabelPatches(); break;
                case NORM64: figure.cropLabelGrayNormPatches(64, 64); break;
            }

            for (Panel panel : figure.panels)
            {
                if (panel.labelRect == null) continue; //No label rect
                if (panel.panelLabel.length() != 1) continue; //Label has more than 1 char, we ignore for now.

                String name = PanelSeg.getLabelCharFolderName(panel.panelLabel.charAt(0));
                Path folder = typeFolder.resolve(name);

                String[] folderWords = imageFile.split("\\\\");
                name = folderWords[folderWords.length - 2] + "-" + folderWords[folderWords.length - 1] + "-";
                name += panel.labelRect.toString() + ".png";

                Path file = folder.resolve(name);

                switch (type)
                {
                    case ORIGINAL: opencv_imgcodecs.imwrite(file.toString(), panel.labelPatch); break;
                    case NORM64: opencv_imgcodecs.imwrite(file.toString(), panel.labelGrayNormPatch); break;
                }

            }

        }
    }
}
