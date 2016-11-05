package gov.nih.nlm.lhc.openi.panelseg;

import libsvm.svm;
import libsvm.svm_node;
import org.bytedeco.javacpp.opencv_core;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 * Created by jzou on 11/4/2016.
 */
public class PanelSegLabelRegHogLeNet5 extends PanelSegLabelRegHog
{
    protected static MultiLayerNetwork leNet5Model;

    protected static void initialize() {
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

    public PanelSegLabelRegHogLeNet5()
    {
        super();
    }

    protected void LeNet5Classification()
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

            panel.labelScore = posProb;
        }

        figure.panels = panels;
    }

    protected void MergeRecognitionLabelsSimple()
    {
        if (figure.panels.size() == 0) return;

        ArrayList<Panel> candidates = new ArrayList<>();
        for (int j = 0; j < figure.panels.size(); j++)
        {
            Panel panel = figure.panels.get(j);
            if (panel.labelScore < 0.5) continue;

            candidates.add(panel);
        }

        figure.panels = RemoveOverlappedCandidates(candidates);
    }

    /**
     * The main entrance function to perform segmentation.
     * Call getResult* functions to retrieve result in different format.
     */
    void segment(opencv_core.Mat image)
    {
        //run super class segmentation steps.
        figure = new Figure(image); //Common initializations for all segmentation method.
        HoGDetect();		//HoG Detection, detected patches are stored in hogDetectionResult
        mergeDetectedLabelsSimple();	//All detected patches are merged into figure.panels.

        //additional steps for this method.
        LeNet5Classification();			//SVM classification of each detected patch in figure.panels.
        MergeRecognitionLabelsSimple();
    }
}
