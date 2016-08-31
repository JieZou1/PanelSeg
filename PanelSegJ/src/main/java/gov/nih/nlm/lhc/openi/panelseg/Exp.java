package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.opencv_core;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 /**
 * The base class for all experiment related operations. It works with the list text file under experiment folder <p>
 * This class is not intended to be instantiated, so we make it abstract.
 *
 * Created by jzou on 8/30/2016.
 */
public class Exp {
    protected Path listFile;        //The list file containing the samples to be experimented with
    protected Path targetFolder;    //The folder for saving the result

    protected List<Path> imagePaths;    //The paths to sample images.

    /**
     * Ctor, set targetFolder and then collect all imagePaths
     * It also clean the targetFolder
     *
     * @param listFile
     * @param targetFolder
     */
    protected Exp(String listFile, String targetFolder) {
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

        System.out.println("Total number of imageColor is: " + imagePaths.size());

        AlgMiscEx.createClearFolder(this.targetFolder);
    }

}
