package gov.nih.nlm.lhc.openi.panelseg;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_core.Mat;

/**
 * Created by jzou on 8/25/2016.
 */
public class AnnotationPreview extends Annotation
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 1)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar AnnotationPreview <annotation folder>");
            System.out.println("	This is a utility program to generate images with annotations superimposed on the original figure images.");
            System.exit(0);
        }

        AnnotationPreview preview = new AnnotationPreview(args[0]);
        preview.generatePreview();
        System.out.println("Completed!");
    }

    /**
     * ctor, set annotationFolder and then collect all imagefiles
     * It also clears the preview folder.
     * @param annotationFolder
     */
    AnnotationPreview(String annotationFolder)
    {
        super(annotationFolder);

        //Remove all file in preview folder
        Path preview = this.annotationFolder.resolve("preview");
        try {
            FileUtils.cleanDirectory(preview.toFile());
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
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
            try {	panels = AnnotationiPhotoDraw.loadPanelSeg(annotationFile);			}
            catch (Exception e) {
                System.out.println(e.getMessage());
                load_gt_error = true;
            }

            //System.out.println(Integer.toString(i+1) +  ": Generate Annotation Preview for " + imageFile);

            Mat img = opencv_imgcodecs.imread(imageFile);
            String key = imagePath.getFileName().toString();
            String style = styles.get(key);
            Mat imgAnnotated = load_gt_error ? img.clone() : drawAnnotation(img, panels, style);

            Path preview_path = imagePath.getParent().resolve("preview").resolve(imagePath.getFileName().toString());
            opencv_imgcodecs.imwrite(preview_path.toString(), imgAnnotated);
        }
    }

}
