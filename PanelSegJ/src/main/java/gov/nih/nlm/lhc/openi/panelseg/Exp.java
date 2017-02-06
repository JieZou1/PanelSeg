package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static java.util.concurrent.ForkJoinTask.invokeAll;

/**
 * The base class for all experiment related operations. It works with the list text file under experiment folder <p>
 * This class is not intended to be instantiated, so we make it abstract.
 *
 * Created by jzou on 8/30/2016.
 */
abstract class Exp {
    public enum LabelPreviewType {ORIGINAL, NORM64}

    protected Path listFile;        //The list file containing the samples to be experimented with
    protected Path targetFolder;    //The folder for saving the result

    protected List<Path> imagePaths;    //The paths to sample images.

    //protected Exp() {}

    /**
     * Ctor, set targetFolder and then collect all imagePaths
     * It also clean the targetFolder, if clearTargetFolder is set to true
     *
     * @param listFile
     * @param targetFolder
     * @param clearTargetFolder
     */
    protected Exp(String listFile, String targetFolder, boolean clearTargetFolder) {
        this.targetFolder = Paths.get(targetFolder);
        this.listFile = Paths.get(listFile);

        //Read the sample list
        imagePaths = new ArrayList<>();
        try {
            try (BufferedReader br = new BufferedReader(new FileReader(listFile))) {
                String line;
                while ((line = br.readLine()) != null)
                    imagePaths.add(Paths.get(line));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("Total number of image is: " + imagePaths.size());

        if (clearTargetFolder)
            AlgMiscEx.createClearFolder(this.targetFolder);
    }

    //The method for handling processing of 1 sample, mostly for implementing multi-threading processing in Fork/Join framework
    abstract void doWork(int k) throws Exception;

    /**
     * Helper function for generating panel label patch filenames for saving to disks.
     * The panel label patch filename is in the format of "<set>-<figure filename>-<labelRect>.png".
     * @param figureFile
     * @param panel
     * @return
     */
    protected String getLabelPatchFilename(String figureFile, Panel panel)
    {
        String[] folderWords = figureFile.split("\\\\");
        String name = folderWords[folderWords.length - 2] + "-" + folderWords[folderWords.length - 1] + "-";
        name += panel.labelRect.toString() + ".png";

        return name;
    }

    /**
     * Load GroundTruth annotation from imagePath
     * @param imagePath
     * @return
     */
    protected ArrayList<Panel> loadGtAnnotation(Path imagePath)
    {
        ArrayList<Panel> gtPanels = null;
        String imageFile = imagePath.toString();
        String xmlFile = FilenameUtils.removeExtension(imageFile) + "_data.xml";

        File annotationFile = new File(xmlFile);
        try {
            gtPanels = iPhotoDraw.loadPanelSegGt(annotationFile);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
        return gtPanels;
    }

    /**
     * Load Auto segmentation result from imagePath
     * @param imagePath
     * @return
     */
    protected ArrayList<Panel> loadAutoAnnotation(Path imagePath)
    {
        ArrayList<Panel> autoPanels = null;
        String imageFile = imagePath.toFile().getName();
        String xmlFile = FilenameUtils.removeExtension(imageFile) + "_data.xml";
        File annotationFile  = targetFolder.resolve(xmlFile).toFile();
        try {
            autoPanels = iPhotoDraw.loadPanelLabelRegAuto(annotationFile);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
        return autoPanels;
    }

    /**
     * Filter out gtPanels, which do not contain qualified panel labels.
     * @param panels
     * @return
     */
    protected ArrayList<Panel> filterLabels(ArrayList<Panel> panels)
    {
        ArrayList<Panel> panelsFiltered = new ArrayList<>();
        for (int i = 0; i < panels.size(); i++) {
            Panel panel = panels.get(i);
            if (panel.labelRect == null || panel.labelRect.isEmpty())
                continue; //In this panel, there is no label.
            if (panel.panelLabel == null || panel.panelLabel.length() != 1)
                continue; //For now, we can handle single char panel label only
            panelsFiltered.add(panel);
        }
        return panelsFiltered;
    }


    protected void saveSegResult(String imageFile, opencv_core.Mat image, List<Panel> panels)
    {
        //Save result in iPhotoDraw XML file
        String xmlFile = FilenameUtils.removeExtension(imageFile) + "_data.xml";
        Path xmlPath = targetFolder.resolve(xmlFile);
        try {
            iPhotoDraw.savePanelSeg(xmlPath.toFile(), panels);
        } catch (Exception e) {
            e.printStackTrace();
        }

        //Save original jpg file
        Path origPath = targetFolder.resolve(imageFile);
        opencv_imgcodecs.imwrite(origPath.toString(), image);

        //Save preview in jpg file
        Path previewFolder = targetFolder.resolve("preview");
        if (!Files.exists(previewFolder)) previewFolder.toFile().mkdir();

        Path previewPath = previewFolder.resolve(imageFile);
        opencv_core.Mat preview = Figure.drawAnnotation(image, panels);
        opencv_imgcodecs.imwrite(previewPath.toString(), preview);
    }

    /**
     * Entry function
     */
    protected void doWorkSingleThread() throws Exception
    {
        long startTime = System.currentTimeMillis();
        for (int k = 0; k < imagePaths.size(); k++) doWork(k);
        long endTime = System.currentTimeMillis();

        System.out.println("Total processing time: " + (endTime - startTime)/1000.0 + " seconds.");
        System.out.println("Average processing time: " + ((endTime - startTime)/1000.0)/imagePaths.size() + " seconds.");
    }

    protected void doWorkMultiThread()
    {
        ExpTask[] tasks = ExpTask.createTasks(this, imagePaths.size(), 4);
        invokeAll(tasks);
//        ExpTask task = new ExpTask(this, 0, imagePaths.size());
//        task.invoke();
    }
}

