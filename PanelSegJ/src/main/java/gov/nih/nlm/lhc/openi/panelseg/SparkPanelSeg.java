package gov.nih.nlm.lhc.openi.panelseg;

import libsvm.svm_model;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.nio.file.Path;
import java.nio.file.Paths;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

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

        //panelSeg(args);
        test(args);
    }

    static void panelSeg(String[] args) throws Exception
    {
        final SparkConf sparkConf = new SparkConf().setAppName("Panel Segmentation");
        final JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Clear and broadcast targetFolder
        Path targetFolder = Paths.get("/hadoop/storage/user/jzou/projects/PanelSeg/Exp/eval");
        AlgMiscEx.createClearFolder(targetFolder);
        AlgMiscEx.createClearFolder(targetFolder.resolve("preview"));
        System.out.println(targetFolder.toString() + "is cleaned!");

        Broadcast<String> TargetFolder = sc.broadcast(targetFolder.toString());
        System.out.println("TargetFolder is broad casted: " + targetFolder);

        //Set and broadcast method
        PanelSeg.Method method = PanelSeg.Method.LabelRegHogLeNet5SvmBeam;
        Broadcast<PanelSeg.Method> Method = sc.broadcast(method);
        System.out.println("Method is broad casted: " + method);

        //Set and broadcast LabelDetectHog models
        String propLabelSetsHOG = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz123456789";
        String propLabelHogModel = "svmModel_19409_17675";
        LabelDetectHog.initialize(propLabelSetsHOG, propLabelHogModel);
        System.out.println("LabelDetectHog model is loaded!");

        Broadcast<String[]> LabelDetectHog_labelSetsHOG = sc.broadcast(LabelDetectHog.labelSetsHOG);
        Broadcast<float[][]> LabelDetectHog_models = sc.broadcast(LabelDetectHog.models);
        System.out.println("LabelDetectHog model is broad-casted!");

        //Load and broadcast LabelClassifyHogSvm model
        String propSvmModel = "svm_model_rbf_8.0_0.03125";
        LabelClassifyHogSvm.initialize(propSvmModel);
        System.out.println("LabelClassifyHogSvm model is loaded!");

        Broadcast<svm_model> LabelClassifyHogSvm_svmModel = sc.broadcast(LabelClassifyHogSvm.svmModel);
        System.out.println("LabelClassifyHogSvm model is broad-casted!");

        //Load and broadcast LabelSequenceClassify models
        String propLabelSeqSvmModels = "svm_model_2_2048.0_8.0,scaling2.txt;svm_model_3_2048.0_8.0,scaling3.txt;svm_model_4_512.0_8.0,scaling4.txt;svm_model_5_128.0_8.0,scaling5.txt;svm_model_6_32.0_0.5,scaling6.txt";
        LabelSequenceClassify.initialize(propLabelSeqSvmModels);
        System.out.println("LabelSequenceClassify models are loaded!");

        Broadcast<svm_model[]> LabelSequenceClassify_svmModels = sc.broadcast(LabelSequenceClassify.svmModels);
        Broadcast<float[][]> LabelSequenceClassify_mins = sc.broadcast(LabelSequenceClassify.mins);
        Broadcast<float[][]> LabelSequenceClassify_ranges = sc.broadcast(LabelSequenceClassify.ranges);
        System.out.println("LabelSequenceClassify models are broad-casted!");

        //Load and broad cast LabelClassifyLeNet5 model
        String propLabelLeNet5Model = "LeNet5-28-23500_25130.model";
        LabelClassifyLeNet5.initialize(propLabelLeNet5Model);
        System.out.println("LabelClassifyLeNet5 model is loaded!");

        Broadcast<MultiLayerNetwork> LabelClassifyLeNet5_leNet5Model = sc.broadcast(LabelClassifyLeNet5.leNet5Model);
        Broadcast<String> LabelClassifyLeNet5_propLabelLeNet5Model = sc.broadcast(propLabelLeNet5Model);
        System.out.println("LabelClassifyLeNet5 model is broad-casted!");

        //Processing images
        JavaRDD<String> lines = sc.textFile(args[0]);
        lines.foreach(new SparkPanelSegFunc(
                TargetFolder,
                Method,
                LabelDetectHog_labelSetsHOG,
                LabelDetectHog_models,
                LabelClassifyHogSvm_svmModel,
                LabelSequenceClassify_svmModels,
                LabelSequenceClassify_mins,
                LabelSequenceClassify_ranges,
                LabelClassifyLeNet5_leNet5Model,
                LabelClassifyLeNet5_propLabelLeNet5Model));

        System.out.println("Completed!");
    }

    static void test(String[] args) throws Exception
    {
        final SparkConf sparkConf = new SparkConf().setAppName("Test");
        final JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Clear and broadcast targetFolder
        Path targetFolder = Paths.get("/hadoop/storage/user/jzou/projects/PanelSeg/Exp/eval");
        AlgMiscEx.createClearFolder(targetFolder);
        AlgMiscEx.createClearFolder(targetFolder.resolve("preview"));
        System.out.println(targetFolder.toString() + "is cleaned!");

        //Processing images
        JavaRDD<String> lines = sc.textFile(args[0]);
        lines.foreach(new SparkTestFunc());

        System.out.println("Completed!");
    }

}

