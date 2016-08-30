package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

/**
 * Created by jzou on 8/30/2016.
 */
public class AnnotationLabelPreview extends Annotation
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 1)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar AnnotationLabelPreview <annotation folder>");
            System.out.println("	This is a utility program to generate image preview of label rects.");
            System.exit(0);
        }

        AnnotationLabelPreview preview = new AnnotationLabelPreview(args[0]);
        preview.generatePreview();
        System.out.println("Completed!");
    }

    /**
     * ctor, set annotationFolder and then collect all imagefiles
     * It also clears the LabelPreview folder.
     * @param annotationFolder
     */
    AnnotationLabelPreview(String annotationFolder)
    {
        super(annotationFolder);

        Path preview = this.annotationFolder.resolve("LabelPreview");
        if (!Files.exists(preview)) preview.toFile().mkdir();

        //Remove all file in preview folder
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
            ArrayList<Panel> panels = null;
            try
            {
                panels = AnnotationiPhotoDraw.loadPanelSeg(annotationFile);
            }
            catch (Exception e) {
                System.out.println(e.getMessage());
            }

            //System.out.println(Integer.toString(i+1) +  ": Generate Annotation Preview for " + imageFile);

            opencv_core.Mat img = opencv_imgcodecs.imread(imageFile);
            //opencv_core.Mat imgAnnotated = drawAnnotation(img, panels, style);

            Path preview_path = imagePath.getParent().resolve("preview").resolve(imagePath.getFileName().toString());
            //opencv_imgcodecs.imwrite(preview_path.toString(), imgAnnotated);
        }
    }
}
