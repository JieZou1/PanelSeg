package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Classify Labels using LeNet5
 * Refactored from PanelSegLabelRegHogLeNet5 (which is obsolete) to make class hierarchy simpler.
 *
 * Created by jzou on 11/7/2016.
 */
public class LabelClassifyLeNet5
{
    //region Static variables (leNet5Model), need to be initialized once by initialize() function
    protected static MultiLayerNetwork leNet5Model;

    static void initialize() {
        if (leNet5Model == null) {
            try {
                //log.info("Load Models...");
                String modelFile = "LeNet5.model";
                leNet5Model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
                System.out.println(modelFile + " is loaded.");
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }

    //endregion

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
            double posProb = letNet5Probs.getDouble(i, 1);

            panel.labelScore = panel.labelClassifyPosLetNet5Score = posProb;
        }

        figure.panels = panels;
    }

    /**
     * Remove false alarms based on LeNet5 classification result (pos_prob is less than 0.5)
     */
    void removeFalseAlarms()
    {
        if (figure.panels.size() == 0) return;

        ArrayList<Panel> candidates = new ArrayList<>();
        for (int j = 0; j < figure.panels.size(); j++)
        {
            Panel panel = figure.panels.get(j);
            if (panel.labelScore < 0.5) continue;

            candidates.add(panel);
        }

        figure.panels = PanelSeg.RemoveOverlappedLabelCandidates(candidates);
    }

}
