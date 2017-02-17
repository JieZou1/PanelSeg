package gov.nih.nlm.lhc.openi.panelseg;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_objdetect;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Properties;

import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 * Label classification using SVM classifier on HoG features
 * Refactored from PanelSegLabelRegHogSvm (which is obsolete) to make class hierarchy simpler.
 *
 * Created by jzou on 11/7/2016.
 */
final class LabelClassifyHogSvm
{
    protected static final Logger log = LoggerFactory.getLogger(LabelClassifyHogSvm.class);

    private static svm_model svmModel;
    private static String propSvmModel;

    static void initialize(Properties properties) throws Exception
    {
        propSvmModel = properties.getProperty("LabelSvmModel");
        if (propSvmModel == null) throw new Exception("ERROR: LabelSvmModel property is Missing.");

        switch (propSvmModel)
        {
            case "svm_model_linear_0.5_94":
            case "svm_model_rbf_32.0_0.0078125_96.3":
            case "svm_model_rbf_8.0_0.03125":
                InputStream modelStream = LabelClassifyHogSvm.class.getClassLoader().getResourceAsStream(propSvmModel);
                BufferedReader br = new BufferedReader(new InputStreamReader(modelStream));
                svmModel = libsvm.svm.svm_load_model(br);
                log.info(propSvmModel + " is loaded. nr_class is " + svmModel.nr_class);
                break;
            default:
                throw new Exception("LabelSvmModel of " + propSvmModel + "is Unknown!");
        }
    }

    private Figure figure;
    private opencv_objdetect.HOGDescriptor hog;

    LabelClassifyHogSvm(Figure figure)
    {
        this.figure = figure;

        opencv_core.Size winSize_64 = new opencv_core.Size(64, 64);
        //static private Size winSize_32 = new Size(32, 32); //The size of the training label patches
        opencv_core.Size blockSize = new opencv_core.Size(16, 16);
        opencv_core.Size blockStride = new opencv_core.Size(8, 8);
        opencv_core.Size cellSize = new opencv_core.Size(8, 8);
        int nbins = 9;
//      int derivAperture = 1;
//      double winSigma = -1;
//      double L2HysThreshold = 0.2;
//      boolean gammaCorrection = true;
//      int nLevels = 64;

        hog = new opencv_objdetect.HOGDescriptor(winSize_64, blockSize, blockStride, cellSize, nbins);
        //hog = new HOGDescriptor(winSize_32, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, _histogramNormType, _L2HysThreshold, gammaCorrection, nlevels, _signedGradient)
    }

    /**
     * Extract HoG descriptors from gray patch
     * @param grayPatch
     * @return
     */
    float[] featureExtraction(opencv_core.Mat grayPatch)
    {
        FloatPointer descriptors = new FloatPointer();
        hog.compute(grayPatch, descriptors);
        //hog.compute(grayPatch, descriptors, winStride, padding, null);

        int n = (int)hog.getDescriptorSize();
        float[] features = new float[n];		descriptors.get(features);
        return features;
    }

    void svmClassification() throws Exception
    {
        if (figure.panels.size() == 0) return;

        for (int i = 0; i < figure.panels.size(); i++)
        {
            Panel panel = figure.panels.get(i);
            if (panel == null) continue;

            opencv_core.Mat patch = AlgOpenCVEx.cropImage(figure.imageGray, panel.labelRect);
            opencv_core.Mat patchNormalized = new opencv_core.Mat();
            resize(patch, patchNormalized, hog.winSize());

            float[] feature = featureExtraction(patchNormalized);
            svm_node[] svmNode = LibSvmEx.float2SvmNode(feature);
            double[] svmProbs = new double[LibSvmEx.getNrClass(svmModel)];
	        /*double label = */svm.svm_predict_probability(svmModel, svmNode, svmProbs);

            panel.labelProbs = svmProbs;
        }
    }

    void svmClassificationWithLeNet5() throws Exception
    {
        if (figure.panels.size() == 0) return;

        for (int i = 0; i < figure.panels.size(); i++)
        {
            Panel panel = figure.panels.get(i);
            if (panel == null) continue;

            opencv_core.Mat patch = AlgOpenCVEx.cropImage(figure.imageGray, panel.labelRect);
            opencv_core.Mat patchNormalized = new opencv_core.Mat();
            resize(patch, patchNormalized, hog.winSize());

            float[] feature = featureExtraction(patchNormalized);
            svm_node[] svmNode = LibSvmEx.float2SvmNode(feature);
            double[] svmProbs = new double[LibSvmEx.getNrClass(svmModel)];
	        /*double label = */svm.svm_predict_probability(svmModel, svmNode, svmProbs);

            double negProb = panel.labelClassifyLetNet5Probs[0];
            double posProb = panel.labelClassifyLetNet5Probs[1];

            double[] probs = new double[PanelSeg.labelChars.length + 1]; //We add 1 more for negative class.
            for (int j = 0; j < PanelSeg.labelChars.length; j++)
                probs[j] = posProb * svmProbs[j];
            probs[PanelSeg.labelChars.length] = negProb;
            panel.labelProbs = probs;
        }
    }

    void mergeRecognitionLabelsSimple()
    {
        if (figure.panels.size() == 0) return;

        //set label and score according to the max of labelProbs, computed by SVM
        ArrayList<Panel> candidates = new ArrayList<>();
        for (int j = 0; j < figure.panels.size(); j++)
        {
            Panel panel = figure.panels.get(j);
            int maxIndex = AlgMiscEx.findMaxIndex(panel.labelProbs);
            if (maxIndex == PanelSeg.labelChars.length) continue; //Classified as a negative sample.

            panel.labelScore = panel.labelProbs[maxIndex];
            panel.panelLabel = "" + PanelSeg.labelChars[maxIndex];
            candidates.add(panel);
        }

        figure.panels = PanelSeg.RemoveOverlappedLabelCandidates(candidates);
    }

    /**
     * Use simple thresholding to pick the final result
     * @param threshold
     */
    void threshold(double threshold)
    {
        //additional steps for this method.
        if (figure.panels.size() == 0) return;

        //set label and score according to the max of labelProbs, computed by SVM
        ArrayList<Panel> candidates = new ArrayList<>();
        for (int j = 0; j < figure.panels.size(); j++)
        {
            Panel panel = figure.panels.get(j);
            if (panel.labelScore < threshold) continue;
            candidates.add(panel);
        }

        figure.panels = candidates;

    }

}
