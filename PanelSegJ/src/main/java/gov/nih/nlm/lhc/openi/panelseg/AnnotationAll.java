package gov.nih.nlm.lhc.openi.panelseg;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

/**
 * The base class for all annotation related operations on all data sets. <p>
 * This class is not intended to be instantiated, so we make it abstract.
 *
 * Created by jzou on 8/26/2016.
 */
public abstract class AnnotationAll
{
    protected Path dataFolder;
    protected ArrayList<Path> annotationFolders;

    /**
     * ctor, set dataFolder and then collect all annotationFolders
     * @param dataFolder
     */
    protected AnnotationAll(String dataFolder)
    {
        this.dataFolder = Paths.get(dataFolder);
        annotationFolders = AlgorithmEx.CollectSubfolders(this.dataFolder);
    }

}
