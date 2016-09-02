package gov.nih.nlm.lhc.openi.panelseg;

import java.nio.file.Path;

/**
 * Created by jzou on 8/26/2016.
 */
public class DataPreviewAll extends DataAll
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 1)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar gov.nih.nlm.lhc.openi.panelseg.DataPreviewAll <data folder>");
            System.out.println("	This is a utility program to generate images with annotations superimposed on the original figure images.");
            System.exit(0);
        }

        DataPreviewAll statistics = new DataPreviewAll(args[0]);
        statistics.generatePreview();
        System.out.println("Completed!");
    }

    /**
     * ctor, set dataFolder and then collect all setFolders
     * @param dataFolder
     */
    private DataPreviewAll(String dataFolder)
    {
        super(dataFolder);
    }

    private void generatePreview()
    {
        for (Path annotation_folder : setFolders)
        {
            System.out.println("Generate Preview for: " + annotation_folder);
            DataPreview preview = new DataPreview(annotation_folder.toString());
            preview.generatePreview();
        }
    }
}
