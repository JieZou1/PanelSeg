package gov.nih.nlm.lhc.openi.panelseg;

import libsvm.svm_model;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * Created by jzou on 4/26/2017.
 */
class SparkPanelSegFunc implements VoidFunction<String>
{
    private Broadcast<String> TargetFolder;

    private Path targetFolder;          //The folder for saving the result
    private PanelSeg.Method method;

    public SparkPanelSegFunc(
            Broadcast<String> TargetFolder,
            Broadcast<PanelSeg.Method> Method,
            Broadcast<String[]> LabelDetectHog_labelSetsHOG,
            Broadcast<float[][]> LabelDetectHog_models,
            Broadcast<svm_model> LabelClassifyHogSvm_svmModel,
            Broadcast<svm_model[]> LabelSequenceClassify_svmModels,
            Broadcast<float[][]> LabelSequenceClassify_mins,
            Broadcast<float[][]> LabelSequenceClassify_ranges,
            Broadcast<MultiLayerNetwork> LabelClassifyLeNet5_leNet5Model,
            Broadcast<String> LabelClassifyLeNet5_propLabelLeNet5Model
    )
    {
        this.TargetFolder = TargetFolder;
        this.method = Method.value();

        LabelDetectHog.labelSetsHOG = LabelDetectHog_labelSetsHOG.value();
        LabelDetectHog.models = LabelDetectHog_models.value();

        LabelClassifyHogSvm.svmModel = LabelClassifyHogSvm_svmModel.value();

        LabelSequenceClassify.svmModels = LabelSequenceClassify_svmModels.value();
        LabelSequenceClassify.mins = LabelSequenceClassify_mins.value();
        LabelSequenceClassify.ranges = LabelSequenceClassify_ranges.value();

        LabelClassifyLeNet5.leNet5Model = LabelClassifyLeNet5_leNet5Model.value();
        LabelClassifyLeNet5.propLabelLeNet5Model = LabelClassifyLeNet5_propLabelLeNet5Model.value();
    }

    @Override
    public void call(String imagePath) throws Exception
    {
        targetFolder = Paths.get(TargetFolder.value());

        String imageFile = Paths.get(imagePath).toFile().getName();
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
