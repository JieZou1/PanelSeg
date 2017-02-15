package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_imgcodecs;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;

/**
 * This is a utility program to generate preview of label patches.
 * It can also be used for generate positive training samples for label detection.
 *
 * Created by jzou on 2/15/2017.
 */
final class ExpGenLabelPatchPreview extends Exp
{
    public static void main(String args[])
    {
        log.info("Crop the Ground-truth Label Patches for Preview purpose.");

        ExpGenLabelPatchPreview exp = new ExpGenLabelPatchPreview();
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

    private String propPreviewType;
    private enum LabelPreviewType {ORIGINAL, NORM64}
    private LabelPreviewType type;
    private Path typeFolder;

    private void setPreviewType() throws Exception
    {
        switch (propPreviewType)
        {
            case "ORIGINAL": type = LabelPreviewType.ORIGINAL; break;
            case "NORM64": type = LabelPreviewType.NORM64; break;
            default: throw new Exception("Unknown Preview Type: " + propPreviewType);
        }
    }

    @Override
    void loadProperties() throws Exception
    {
        loadProperties("ExpGenLabelPatchPreview.properties");

        propThreads = getProperty("Threads");
        propListFile = getProperty("ListFile");
        propTargetFolder = getProperty("TargetFolder");
        propPreviewType = getProperty("PreviewType");

        waitKeyContinueOrQuit("Configuration Okay? Press any key to continue, press N to quit");
    }

    @Override
    void initialize() throws Exception
    {
        setMultiThreading();
        setListFile();
        setTargetFolder(false);
        setPreviewType();

        //Clean up all the folders
        typeFolder = this.targetFolder.resolve(type.toString());
        AlgMiscEx.createClearFolder(typeFolder);
        for (char c : PanelSeg.labelChars)
        {
            String name = PanelSeg.getLabelCharFolderName(c);
            Path folder = typeFolder.resolve(name);
            AlgMiscEx.createClearFolder(folder);
        }
    }

    @Override
    void doWork() throws Exception
    {
        if (threads > 1) doWorkMultiThread();
        else doWorkSingleThread();
    }

    /**
     * Generate label preview
     */
    void doWork(int i) throws Exception
    {
        super.doWork(i);

        Path imagePath = imagePaths.get(i);
        String imageFile = imagePath.toString();

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
        switch (type) {
            case ORIGINAL:
                figure.cropLabelPatches();
                break;
            case NORM64:
                figure.cropLabelGrayNormPatches(64, 64);
                break;
        }

        for (Panel panel : figure.panels) {
            if (panel.labelRect == null) continue; //No label rect
            if (panel.panelLabel.length() != 1) continue; //Label has more than 1 char, we ignore for now.

            String name = PanelSeg.getLabelCharFolderName(panel.panelLabel.charAt(0));
            Path folder = typeFolder.resolve(name);

            String[] folderWords = imageFile.split("\\\\");
            name = folderWords[folderWords.length - 2] + "-" + folderWords[folderWords.length - 1] + "-";
            name += panel.labelRect.toString() + ".png";

            Path file = folder.resolve(name);

            switch (type) {
                case ORIGINAL:
                    opencv_imgcodecs.imwrite(file.toString(), panel.labelPatch);
                    break;
                case NORM64:
                    opencv_imgcodecs.imwrite(file.toString(), panel.labelGrayNormPatch);
                    break;
            }
        }
    }
}
