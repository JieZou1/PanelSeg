package gov.nih.nlm.lhc.openi.panelseg;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;

import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_core.Mat;

/**
 * Created by jzou on 8/25/2016.
 */
final class DataPreview extends Data
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 1)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar DataPreview <annotation folder>");
            System.out.println("	This is a utility program to generate images with annotations superimposed on the original figure images.");
            System.exit(0);
        }

        DataPreview preview = new DataPreview(args[0]);
        preview.generatePreview();
        System.out.println("Completed!");
    }

    /**
     * ctor, set setFolder and then collect all imagefiles
     * It also clears the preview folder.
     * @param annotationFolder
     */
    DataPreview(String annotationFolder) throws Exception
    {
        super(annotationFolder);

        Path preview = this.setFolder.resolve("preview");
        AlgMiscEx.createClearFolder(preview);
    }

    /**
     * Entry function for generating annotation preview
     */
    void generatePreview()
    {
        for (int i = 0; i < imagePaths.size(); i++)
        {
            Path imagePath = imagePaths.get(i);
            String imageFile = imagePath.toString();
            String xmlFile = FilenameUtils.removeExtension(imageFile) + "_data.xml";

            //if (!xmlFile.endsWith("PMC3025345_kcj-40-684-g002_data.xml")) {i++; continue; }

            //Load annotation
            File annotationFile = new File(xmlFile);
            ArrayList<Panel> panels = null; boolean load_gt_error = false;
            try
            {
                panels = iPhotoDraw.loadPanelSegGt(annotationFile);
            }
            catch (Exception e) {
                System.out.println(e.getMessage());
                load_gt_error = true;
            }

            //System.out.println(Integer.toString(i+1) +  ": Generate Data Preview for " + imageFile);

            Mat img = opencv_imgcodecs.imread(imageFile);
            String key = imagePath.getFileName().toString();
            String style = styles.get(key);
            Mat imgAnnotated = load_gt_error ? img.clone() : Figure.drawAnnotation(img, panels, style);

            Path preview_path = imagePath.getParent().resolve("preview").resolve(imagePath.getFileName().toString());
            opencv_imgcodecs.imwrite(preview_path.toString(), imgAnnotated);
        }
    }

}
