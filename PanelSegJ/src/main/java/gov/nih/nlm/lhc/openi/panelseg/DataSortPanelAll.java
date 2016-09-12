package gov.nih.nlm.lhc.openi.panelseg;

import java.nio.file.Path;

/**
 * Created by jzou on 9/2/2016.
 */
final class DataSortPanelAll extends DataAll
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 1)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar gov.nih.nlm.lhc.openi.panelseg.DataSortPanelAll <data folder>");
            System.out.println("	This is a utility program to sort iPhotoDraw annotation.");
            System.out.println("	Make label annotation on top of panel annotation to make manual modification more convenient.");
            System.out.println("	It will overwrite the existing iPhotoDraw annotation file. SO BE CAREFUL!!!");
            System.exit(0);
        }

        DataSortPanelAll sort = new DataSortPanelAll(args[0]);
        sort.sort();
        System.out.println("Completed!");
    }

    /**
     * ctor, set dataFolder and then collect all setFolders
     * @param dataFolder
     */
    private DataSortPanelAll(String dataFolder)
    {
        super(dataFolder);
    }

    private void sort()
    {
        for (Path annotation_folder : setFolders)
        {
            System.out.println("Sort Panel for: " + annotation_folder);
            DataSortPanel preview = new DataSortPanel(annotation_folder.toString());
            preview.sort();
        }
    }
}
