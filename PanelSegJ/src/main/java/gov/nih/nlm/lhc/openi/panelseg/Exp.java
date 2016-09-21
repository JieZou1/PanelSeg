package gov.nih.nlm.lhc.openi.panelseg;

import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

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

    protected Exp() {}

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

    abstract void generate(int k); //The method for handling processing of 1 sample, mostly for implementing multi-threading processing in Fork/Join framework

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
}
