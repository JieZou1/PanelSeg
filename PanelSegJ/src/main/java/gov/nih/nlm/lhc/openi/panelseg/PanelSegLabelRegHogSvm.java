package gov.nih.nlm.lhc.openi.panelseg;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

import java.io.IOException;
import java.util.ArrayList;

/**
 * HOG + SVM method for Label Recognition
 * HOG for label detection and SVM for label classification
 *
 * Created by jzou on 9/15/2016.
 */
public class PanelSegLabelRegHogSvm extends PanelSegLabelRegHog
{
    protected static svm_model svmModel;
    protected static void initialize()
    {
        if (svmModel == null)
        {
            try {
                //String svm_model_file = "svm_model_linear_0.5_94";
                //String svm_model_file = "svm_model_rbf_8.0_0.125";
                String svm_model_file = "svm_model_rbf_32.0_0.0078125_96.3";
                svmModel = svm.svm_load_model(svm_model_file);
                System.out.println(svm_model_file + " is loaded.");
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }

    public PanelSegLabelRegHogSvm()
    {
        super();
    }

    protected void SvmClassification()
    {
        if (figure.panels.size() == 0) return;

        for (int i = 0; i < figure.panels.size(); i++)
        {
            Panel info = figure.panels.get(i);
            if (info == null) continue;

            opencv_core.Mat patch = AlgOpenCVEx.cropImage(figure.imageGray, info.labelRect);
            opencv_core.Mat patchNormalized = new opencv_core.Mat();
            resize(patch, patchNormalized, hog.winSize());

            float[] feature = featureExtraction(patchNormalized);
            svm_node[] svmNode = LibSvmEx.float2SvmNode(feature); double[] probs = new double[LibSvmEx.getNrClass(svmModel)];
	        /*double label = */svm.svm_predict_probability(svmModel, svmNode, probs);

            info.labelProbs = probs;
            //figure.segmentationResult.set(i, info);
        }

//		for (int i = 0; i < figure.segmentationResultIndividualLabel.size(); i++)
//		{
//			ArrayList<PanelSegInfo> infos = figure.segmentationResultIndividualLabel.get(i);
//			if (infos == null) continue;
//			for (int j = 0; j < infos.size(); j++)
//			{
//				PanelSegInfo info = infos.get(j);
//				if (info == null) continue;
//
//				//int x = info.labelRect.x, y = info.labelRect.y, w = info.labelRect.width, h = info.labelRect.height;
//				Mat patch = info.labelInverted ? AlgorithmEx.cropImage(figure.imageGrayInverted, info.labelRect) : AlgorithmEx.cropImage(figure.imageGray, info.labelRect);
//		        Mat patchNormalized = new Mat(); resize(patch, patchNormalized, winSize_64);
//
//		        float[] feature = featureExtraction(patchNormalized);
//		        svm_node[] svmNode = LibSvmEx.float2SvmNode(feature); double[] probs = new double[LibSvmEx.getNrClass(svmModel)];
//		        /*double label = */svm.svm_predict_probability(svmModel, svmNode, probs);
//
//		        info.labelProbs = probs;		        infos.set(j, info);
//			}
//			//figure.segmentationResultIndividualLabel.set(i, infos);
//		}
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

        //additional steps for this method.
        SvmClassification();			//SVM classification of each detected patch in figure.panels.
        MergeRecognitionLabelsSimple();
    }

}
