package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import static java.util.concurrent.ForkJoinTask.invokeAll;

/**
 * The base class for all experiment related operations. It works with the list text file under experiment folder <p>
 * This class is not intended to be instantiated, so we make it abstract.
 *
 * Created by jzou on 8/30/2016.
 */
abstract class Exp
{
    protected static final Logger log = LoggerFactory.getLogger(Exp.class);

    public enum LabelPreviewType {ORIGINAL, NORM64}

    protected Properties properties = null;
    protected String propThreads, propListFile, propTargetFolder, propTrainFolder, propTargetFile, propTestFolder;

    protected Path listFile;        //The list file containing the samples to be experimented with
    protected Path trainFolder;     //The folder where training data are loaded from.
    protected Path targetFolder;    //The folder for saving the result
    protected Path targetFile;      //The file for saving the result
    protected Path testFolder;      //The folder where test data are loaded from.

    protected List<Path> imagePaths;    //The paths to sample images.
    protected int threads;   // == 1 Single threading; > 1 multi-threading

    protected void setMultiThreading() throws Exception
    {
        threads = Integer.parseInt(propThreads);
    }
    protected void setListFile()throws Exception
    {
        listFile = Paths.get(propListFile);
        loadListFile();
    }
    protected void setTargetFolder(boolean toClean) throws Exception
    {
        targetFolder = Paths.get(propTargetFolder);
        if (toClean)
        {
            System.out.println(targetFolder + " is going to be cleaned!");

            waitKeyContinueOrQuit("Press any key to continue, press N to quit");

            AlgMiscEx.createClearFolder(targetFolder);
            log.info("Folder " + targetFolder + " is created or cleaned!");
        }
    }
    protected void setTrainFolder() {trainFolder = Paths.get(propTrainFolder);}

    protected void setTargetFile() {targetFile = Paths.get(propTargetFile);}
    protected void setTestFolder() {testFolder = Paths.get(propTestFolder);}

    Exp() {}

    private void loadListFile() throws Exception
    {
        File list_file = listFile.toFile();
        if (!list_file.exists()) throw new Exception("ERROR: " + listFile.toString() + " does not exist.");
        if (!list_file.isFile()) throw new Exception("ERROR: " + listFile.toString() + " is not a file.");

        imagePaths = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(listFile.toFile()))) {
            String line;
            while ((line = br.readLine()) != null)
                imagePaths.add(Paths.get(line));
        }
        log.info("Total number of image is: " + imagePaths.size());
    }

    protected String getProperty(String propName) throws Exception
    {
        String prop = properties.getProperty(propName);
        if (prop == null) throw new Exception("ERROR: " + propName + " property is Missing.");
        log.info(propName + ": " + prop);

        return prop;
    }

    /**
     * Ctor, set targetFolder and then collect all imagePaths
     * It also clean the targetFolder, if clearTargetFolder is set to true
     *
     * @param listFile
     * @param targetFolder
     * @param clearTargetFolder
     */
    protected Exp(String listFile, String targetFolder, boolean clearTargetFolder) throws Exception
    {
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

    abstract void loadProperties() throws Exception;

    protected void loadProperties(String propertiesFile) throws Exception
    {
        properties = new Properties();
        properties.load(this.getClass().getClassLoader().getResourceAsStream(propertiesFile));
    }

    abstract void initialize()  throws  Exception;

    protected void waitKeyContinueOrQuit(String message) throws Exception
    {
        System.out.println();
        System.out.println(message);

        Scanner sc = new Scanner(System.in);
        String s = sc.nextLine();
        if (Objects.equals(s, "n") || Objects.equals(s, "N"))
            throw new Exception("User stops the process!");
    }

    /**
     * Initialization before doWork
     * @throws Exception
     */
    void initialize(String propertyFile) throws Exception
    {
        //Load properties
        properties = new Properties();
        properties.load(this.getClass().getClassLoader().getResourceAsStream(propertyFile));
    }

    /**
     * Do work,
     * call doWorkSingleThread to do work in single thread
     * call doWorkMultiThread to do work in multiple threads
     * @throws Exception
     */
    abstract void doWork() throws Exception;

    //The method for handling processing of 1 sample, mostly for implementing multi-threading processing in Fork/Join framework

    /**
     * Do individual work
     * @param k
     * @throws Exception
     */
    void doWork(int k) throws Exception
    {
        log.info(Integer.toString(k) +  ": processing " + imagePaths.get(k).toString());
    }

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

    protected void doWorkSingleThread() throws Exception
    {
        long startTime = System.currentTimeMillis();
        for (int k = 0; k < imagePaths.size(); k++) doWork(k);
        long endTime = System.currentTimeMillis();

        log.info("Total processing time: " + (endTime - startTime)/1000.0 + " seconds.");
        log.info("Average processing time per image: " + ((endTime - startTime)/1000.0)/imagePaths.size() + " seconds.");
    }

    protected void doWorkMultiThread()
    {
        long startTime = System.currentTimeMillis();
        ExpTask[] tasks = ExpTask.createTasks(this, imagePaths.size(), threads);
        invokeAll(tasks);
//        ExpTask task = new ExpTask(this, 0, imagePaths.size());
//        task.invoke();
        long endTime = System.currentTimeMillis();

        log.info("Total processing time: " + (endTime - startTime)/1000.0 + " seconds.");
        log.info("Average processing time per image: " + ((endTime - startTime)/1000.0)/imagePaths.size() + " seconds.");
    }
}

