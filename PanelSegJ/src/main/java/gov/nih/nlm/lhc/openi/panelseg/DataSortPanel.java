package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;

/**
 * Created by jzou on 8/26/2016.
 */
public class DataSortPanel extends Data
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 1)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar DataSortPanel <annotation folder>");
            System.out.println("	This is a utility program to rewrite iPhotoDraw annotation.");
            System.out.println("	Make label annotation on top of panel annotation to manual modification more convenient.");
            System.out.println("	It will overwrite the existing iPhotoDraw annotation file. SO BE CAREFUL!!!");
            System.exit(0);
        }

        DataSortPanel rewrite = new DataSortPanel(args[0]);
        rewrite.rewrite();
        System.out.println("Completed!");
    }

    /**
     * Ctor, set setFolder and then collect all imagePaths
     * It also load the style annotation into styles.
     * If style.txt is not found in the setFolder, styles is set to null.
     *
     * @param annotationFolder
     */
    protected DataSortPanel(String annotationFolder) {
        super(annotationFolder);
    }

    /**
     * Entry function for rewrite annotation
     */
    void rewrite()
    {
        for (int i = 0; i < imagePaths.size(); i++)
        {
            //if (i < 124) continue;

            Path imagePath = imagePaths.get(i);
            String imageFile = imagePath.toString();
            String xmlFile = FilenameUtils.removeExtension(imageFile) + "_data.xml";

            //if (!xmlFile.endsWith("PMC3025345_kcj-40-684-g002_data.xml")) {i++; continue; }

            //Load annotation
            File annotationFile = new File(xmlFile);
            ArrayList<Panel> panels = null; boolean load_gt_error = false;
            try
            {
                panels = iPhotoDraw.loadPanelSeg(annotationFile);
            }
            catch (Exception e) {
                System.out.println(e.getMessage());
                load_gt_error = true;
            }

            //System.out.println(Integer.toString(i+1) +  ": Rewrite Data for " + imageFile);
            if (!load_gt_error)
            {
                try
                {
                    iPhotoDraw.savePanelSeg(annotationFile, panels);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

}
