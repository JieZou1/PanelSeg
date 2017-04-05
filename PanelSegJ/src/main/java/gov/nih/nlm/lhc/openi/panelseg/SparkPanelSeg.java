package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.api.java.function.VoidFunction2;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.deeplearning4j.util.ModelSerializer;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Properties;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;

/**
 * Panel Segmentation method in Spark
 *
 * Created by jzou on 3/24/2017.
 */
public class SparkPanelSeg
{
    public static void main(String[] args) throws Exception
    {
        if (args.length != 1)
        {
            System.out.println("Usage: SparkPanelSeg <input path>");
            System.exit(-1);
        }

        PanelSeg.Method method = PanelSeg.Method.LabelDetHog;

        Path targetFolder = Paths.get("/hadoop/storage/user/jzou/projects/PanelSeg/Exp/eval");
        AlgMiscEx.createClearFolder(targetFolder);
        AlgMiscEx.createClearFolder(targetFolder.resolve("preview"));

        Properties properties = new Properties();
        properties.load(SparkPanelSeg.class.getResourceAsStream("ExpPanelSeg.properties"));

        final SparkConf sparkConf = new SparkConf().setAppName("PanelSegJ");
        //sparkConf.setMaster("");
        final JavaSparkContext sc = new JavaSparkContext(sparkConf);

        JavaRDD<String> lines = sc.textFile(args[0]);

        lines.foreach(new SparkPanelSegFunc(method, properties, targetFolder));

        System.out.println("Completed!");
    }
}

class SparkPanelSegFunc implements VoidFunction<String>
{
    private Properties properties = null;

    private PanelSeg.Method method;
    private Path targetFolder;    //The folder for saving the result

    public SparkPanelSegFunc(PanelSeg.Method method, Properties properties, Path targetFolder)
    {
        this.method = method;
        this.properties = properties;
        this.targetFolder = targetFolder;
    }

    @Override
    public void call(String imagePath) throws Exception
    {
        String imageFile = Paths.get(imagePath).toFile().getName();

//        method = PanelSeg.Method.LabelDetHog;

//        properties = new Properties();
//        properties.load(this.getClass().getClassLoader().getResourceAsStream("ExpPanelSeg.properties"));
        PanelSeg.initialize(method, properties);

//        LabelDetectHog.labelSetsHOG = new String[1];
//        LabelDetectHog.labelSetsHOG[0] = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz123456789";
//        LabelDetectHog.models = new float[1][];
//        LabelDetectHog.models[0] = LabelDetectHogModels_AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz123456789.svmModel_19409_17675;

        opencv_core.Mat image = imread(imagePath, CV_LOAD_IMAGE_COLOR);
        List<Panel> panels = PanelSeg.segment(image, method);
        saveSegResult(imageFile, image, panels);
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

//class CopyAsBinaFile implements VoidFunction<String>
//{
//    @Override
//    public void call(String imagePath) throws Exception
//    {
//        Path srcPath = Paths.get(imagePath);
//        byte[] content = Files.readAllBytes(srcPath);
//
//        String imageFile = Paths.get(imagePath).toFile().getName();
//        Path dstPath = Paths.get("file://lhce-hadoop/hadoop/storage/user/jzou/projects/PanelSeg/Exp/eval").resolve(imageFile);
//        //Path dstPath = Paths.get(imageFile);
//        Files.write(dstPath, content);
//    }
//}

