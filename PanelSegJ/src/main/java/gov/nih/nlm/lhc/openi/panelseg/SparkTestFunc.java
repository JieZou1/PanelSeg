package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FilenameUtils;
import org.apache.spark.api.java.function.VoidFunction;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.CV_8U;
import static org.bytedeco.javacpp.opencv_core.CV_8UC3;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

/**
 * Created by jzou on 4/26/2017.
 */
class SparkTestFunc implements VoidFunction<String>
{
    private Path targetFolder;
    private PanelSeg.Method method;

    @Override
    public void call(String imagePath) throws Exception
    {
        targetFolder = Paths.get("/hadoop/storage/user/jzou/projects/PanelSeg/Exp/eval");
        //method = PanelSeg.Method.LabelDetHog;

        String imageFile = Paths.get(imagePath).toFile().getName();
        //opencv_core.Mat image = imread(imagePath, CV_LOAD_IMAGE_COLOR);
        opencv_core.Mat image = null;
        for (int i = 0; i < 100; i++)
        {
            image = new opencv_core.Mat(1000, 1000, CV_8UC3);
            rectangle(image, new opencv_core.Rect(100, 100, 100, 100), new opencv_core.Scalar(0, 0, 255, 0));
        }

        Path origPath = targetFolder.resolve(imageFile);
        opencv_imgcodecs.imwrite(origPath.toString(), image);

//        List<Panel> panels = PanelSeg.segment(image, method);
//        saveSegResult(imageFile, image, panels);
    }

    private void saveSegResult(String imageFile, opencv_core.Mat image, List<Panel> panels)
    {
        //Save result in iPhotoDraw XML file
        String xmlFile = FilenameUtils.removeExtension(imageFile) + "_data.xml";
        Path xmlPath = targetFolder.resolve(xmlFile);
        try {
            iPhotoDraw.savePanelSeg(xmlPath.toFile(), panels);
        } catch (Exception e) {
            e.printStackTrace();
        }

        //Save original jpg file
        Path origPath = targetFolder.resolve(imageFile);
        opencv_imgcodecs.imwrite(origPath.toString(), image);

        //Save preview in jpg file
        Path previewFolder = targetFolder.resolve("preview");
        if (!Files.exists(previewFolder)) previewFolder.toFile().mkdir();

        Path previewPath = previewFolder.resolve(imageFile);
        opencv_core.Mat preview = Figure.drawAnnotation(image, panels);
        opencv_imgcodecs.imwrite(previewPath.toString(), preview);
    }

}

