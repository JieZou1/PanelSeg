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
abstract class DataAll
{
    protected Path dataFolder;
    protected ArrayList<Path> setFolders;

    /**
     * ctor, set dataFolder and then collect all setFolders
     * @param dataFolder
     */
    protected DataAll(String dataFolder) throws Exception
    {
        this.dataFolder = Paths.get(dataFolder);
        setFolders = AlgMiscEx.collectSubfolders(this.dataFolder);
    }

}
