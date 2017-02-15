package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Properties;

/**
 * Classify Labels using LeNet5
 * Refactored from PanelSegLabelRegHogLeNet5 (which is obsolete) to make class hierarchy simpler.
 *
 * Created by jzou on 11/7/2016.
 */
final class LabelClassifyLeNet5
{
    protected static final Logger log = LoggerFactory.getLogger(LabelClassifyHogSvm.class);

    private static MultiLayerNetwork leNet5Model = null;
    private static String propLabelLeNet5Model;

    static void initialize(Properties properties) throws Exception
    {
        propLabelLeNet5Model = properties.getProperty("LabelLeNet5Model");
        if (propLabelLeNet5Model == null) throw new Exception("ERROR: LabelLeNet5Model property is Missing.");

        InputStream modelStream = LabelClassifyLeNet5.class.getClassLoader().getResourceAsStream(propLabelLeNet5Model);
        leNet5Model = ModelSerializer.restoreMultiLayerNetwork(modelStream);
        log.info(propLabelLeNet5Model + " is loaded.");
    }

    private Figure figure;

    LabelClassifyLeNet5(Figure figure) {this.figure = figure;}

    /**
     * Classify all label candidates with LeNet5 model
     */
    void LeNet5Classification()
    {
        if (figure.panels.size() == 0) return;

        //Collect all candidate panels and patches
        List<Panel> panels = new ArrayList<>();       List<opencv_core.Mat> patches = new ArrayList<>();
        for (int i = 0; i < figure.panels.size(); i++)
        {
            Panel panel = figure.panels.get(i);            if (panel == null) continue;

            panels.add(panel);
            opencv_core.Mat patch = AlgOpenCVEx.cropImage(figure.imageGray, panel.labelRect);
            patches.add(patch);
        }

        //LeNet5 classification
        NativeImageLoader imageLoader = new NativeImageLoader(28, 28, 1);
        List<INDArray> slices = new ArrayList<>();
        for (int i = 0; i < patches.size(); i++) {
            try
            {
                INDArray arr = imageLoader.asMatrix(patches.get(i));
                slices.add(arr);
            }
            catch (Exception ex) {}
        }
        INDArray imageSet = new NDArray(slices, new int[]{patches.size(), 1, 28, 28});
        imageSet.divi(255.0);
        INDArray letNet5Probs = leNet5Model.output(imageSet);

        //set score to the result from LeNet5 classification
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            panel.labelClassifyLetNet5Probs = new double[2];
            if (Objects.equals(propLabelLeNet5Model, "LeNet5-28-26039_23501.model"))
            {
                panel.labelClassifyLetNet5Probs[0] = letNet5Probs.getDouble(i, 0); //neg
                panel.labelClassifyLetNet5Probs[1] = letNet5Probs.getDouble(i, 1); //pos
            }
            else if (Objects.equals(propLabelLeNet5Model, "LeNet5-28-23500_25130.model"))
            {
                panel.labelClassifyLetNet5Probs[0] = letNet5Probs.getDouble(i, 1); //neg
                panel.labelClassifyLetNet5Probs[1] = letNet5Probs.getDouble(i, 0); //pos
            }
            double posProb = panel.labelClassifyLetNet5Probs[1];

            panel.labelScore = posProb;
        }

        figure.panels = panels;
    }

    /**
     * Remove false alarms based on LeNet5 classification result (pos_prob is less than threshold)
     */
    void removeFalseAlarms(double threshold)
    {
        if (figure.panels.size() == 0) return;

        ArrayList<Panel> candidates = new ArrayList<>();
        for (int j = 0; j < figure.panels.size(); j++)
        {
            Panel panel = figure.panels.get(j);
            if (panel.labelScore < threshold) continue;

            candidates.add(panel);
        }

        figure.panels = PanelSeg.RemoveOverlappedLabelCandidates(candidates);
    }

}
