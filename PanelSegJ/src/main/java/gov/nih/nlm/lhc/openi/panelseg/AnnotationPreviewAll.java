package gov.nih.nlm.lhc.openi.panelseg;

import java.nio.file.Path;

/**
 * Created by jzou on 8/26/2016.
 */
public class AnnotationPreviewAll extends AnnotationAll
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 1)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar AnnotationPreviewAll <data folder>");
            System.out.println("	This is a utility program to generate images with annotations superimposed on the original figure images.");
            System.exit(0);
        }

        AnnotationPreviewAll statistics = new AnnotationPreviewAll(args[0]);
        statistics.generatePreview();
        System.out.println("Completed!");
    }

    /**
     * ctor, set dataFolder and then collect all annotationFolders
     * @param dataFolder
     */
    private AnnotationPreviewAll(String dataFolder)
    {
        super(dataFolder);
    }

    private void generatePreview()
    {
        for (Path annotation_folder : annotationFolders)
        {
            System.out.println("Generate Preview for: " + annotation_folder);
            AnnotationPreview preview = new AnnotationPreview(annotation_folder.toString());
            preview.generatePreview();
        }
    }
}
