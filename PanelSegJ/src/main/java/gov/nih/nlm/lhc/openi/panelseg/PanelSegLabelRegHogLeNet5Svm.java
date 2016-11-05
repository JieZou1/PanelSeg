package gov.nih.nlm.lhc.openi.panelseg;

import libsvm.svm;
import libsvm.svm_model;
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
 * HOG + LeNet5 + SVM method for Label Recognition
 * HOG for label detection
 * LeNet5 for positive (labels) and negative (non-labels) classification
 * SVM for the final label classification
 *
 * Created by jzou on 11/2/2016.
 */
public class PanelSegLabelRegHogLeNet5Svm extends PanelSegLabelRegHogLeNet5
{
    protected static svm_model svmModel;

    protected static void initialize() {
        PanelSegLabelRegHogLeNet5.initialize();

        if (svmModel == null) {
            try {
                String svm_model_file = "svm_model_rbf_8.0_0.03125";
                svmModel = svm.svm_load_model(svm_model_file);
                System.out.println(svm_model_file + " is loaded.");
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }

    public PanelSegLabelRegHogLeNet5Svm()
    {
        super();
    }

    protected void SvmClassification()
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

        //SVM classification
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            double posProb = panel.labelScore;

            opencv_core.Mat patch = patches.get(i);
            opencv_core.Mat patchNormalized = new opencv_core.Mat();
            resize(patch, patchNormalized, hog.winSize());

            float[] feature = featureExtraction(patchNormalized);
            svm_node[] svmNode = LibSvmEx.float2SvmNode(feature);
            double[] svmProbs = new double[LibSvmEx.getNrClass(svmModel)];
	        /*double label = */svm.svm_predict_probability(svmModel, svmNode, svmProbs);

            double[] probs = new double[labelChars.length + 1]; //We add 1 more for negative class.
            for (int j = 0; j < labelChars.length; j++)
                probs[j] = posProb * svmProbs[j];
            probs[labelChars.length] = 1.0 - posProb;
            panel.labelProbs = probs;
        }

        figure.panels = panels;
    }

    protected void MergeRecognitionLabelsSimple()
    {
        if (figure.panels.size() == 0) return;

        //set label and score according to the max of labelProbs, computed by SVM
        ArrayList<Panel> candidates = new ArrayList<>();
        for (int j = 0; j < figure.panels.size(); j++)
        {
            Panel panel = figure.panels.get(j);
            int maxIndex = AlgMiscEx.findMaxIndex(panel.labelProbs);
            if (maxIndex == labelChars.length) continue; //Classified as a negative sample.

            panel.labelScore = panel.labelProbs[maxIndex];
            panel.panelLabel = "" + labelChars[maxIndex];
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

        LeNet5Classification();

        //additional steps for this method.
        SvmClassification();			//SVM classification of each detected patch in figure.panels.
        MergeRecognitionLabelsSimple();
    }

}
