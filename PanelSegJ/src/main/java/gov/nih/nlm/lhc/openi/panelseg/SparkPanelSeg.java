package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.api.java.function.VoidFunction2;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

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

        final SparkConf sparkConf = new SparkConf().setAppName("PanelSegJ");
        //sparkConf.setMaster("");
        final JavaSparkContext sc = new JavaSparkContext(sparkConf);

        JavaRDD<String> lines = sc.textFile(args[0]);

        AlgMiscEx.createClearFolder(Paths.get("./eval"));
        AlgMiscEx.createClearFolder(Paths.get("./eval/preview"));

        lines.foreach(new CopyAsBinaFile());

        System.out.println("Completed!");
    }
}

class CopyAsBinaFile implements VoidFunction<String>
{
    @Override
    public void call(String imagePath) throws Exception
    {
        Path srcPath = Paths.get(imagePath);
        byte[] content = Files.readAllBytes(srcPath);

        String imageFile = Paths.get(imagePath).toFile().getName();
        Path dstPath = Paths.get("/hadoop/storage/user/jzou/projects/PanelSeg/eval").resolve(imageFile);
        //Path dstPath = Paths.get(imageFile);
        Files.write(dstPath, content);
    }
}

class LabelDetHog implements VoidFunction<String>
{
    @Override
    public void call(String imagePath) throws Exception
    {
        PanelSeg.Method method = PanelSeg.Method.LabelDetHog;

        ExpPanelSeg expPanelSeg = new ExpPanelSeg();
        expPanelSeg.targetFolder = Paths.get("./eval");

        LabelDetectHog.labelSetsHOG = new String[1];
        LabelDetectHog.labelSetsHOG[0] = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz123456789";
        LabelDetectHog.models = new float[1][];
        LabelDetectHog.models[0] = LabelDetectHogModels_AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz123456789.svmModel_19409_17675;

        opencv_core.Mat image = imread(imagePath, CV_LOAD_IMAGE_COLOR);
        List<Panel> panels = PanelSeg.segment(image, method);
        expPanelSeg.saveSegResult(imagePath, image, panels);
    }
}
