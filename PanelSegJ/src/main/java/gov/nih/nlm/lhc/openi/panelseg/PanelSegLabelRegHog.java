package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_objdetect.HOGDescriptor;

import java.awt.*;
import java.util.ArrayList;

import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 * Treat Label Recognition as Object Detection using HoG method
 *
 * Created by jzou on 8/31/2016.
 */
class PanelSegLabelRegHog extends PanelSegLabelReg
{
    static final String[] labelSetsHOG = {"AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz123456789"};
//    static final String[] labelSetsHOG = {
//        "A", "a", "B", "bh", "CceG", "D", "d", "E", "F", "f", "gq", "H", "IiJjlrt", "Kk", "L", "Mm", "Nn", "OoQ", "Pp", "R", "Ss", "T", "UuVvWw", "XxYyZz"
//    };
    /**
     * Mapping a Label Char to the LabelSet.
     * @param c
     * @return
     */
    static String getLabelSetName(char c)
    {
        String cStr = Character.toString(c);
        for (String name : labelSetsHOG)
        {
            if (name.indexOf(cStr) >= 0) return name;
        }
        return null;
    }

    protected HOGDescriptor hog;
    protected float[][] svmModels;

    protected ArrayList<ArrayList<Panel>> hogDetectionResult; //The HOG method detection result of all labelSetsToDetect

    /**
     * Constructor, load all SVM models, and initialize the HOGDescriptor
     */
    PanelSegLabelRegHog()
    {
        int n = labelSetsHOG.length;		svmModels = new float[n][];
        for (int i = 0; i < n; i++)
        {
            if (labelSetsHOG[i].equals("AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz123456789"))
                svmModels[i] = PanelSegLabelRegHogModels_AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz123456789.svmModel;

//            String classString = "gov.nih.nlm.lhc.openi.panelseg.PanelSegLabelRegHogModel_" + labelSetsHOG[i];
//            try {
//                Class<?> cls = Class.forName(classString);
//                Field field = cls.getField("svmModel");
//                svmModels[i] = (float[])field.get(null);
//            } catch (Exception e) {
//                // TODO Auto-generated catch block
//                e.printStackTrace();
//            }
        }

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

        hog = new HOGDescriptor(winSize_64, blockSize, blockStride, cellSize, nbins);
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

    protected void HoGDetect()
    {
        int n = labelSetsHOG.length;
        hogDetectionResult = new ArrayList<>();
        for (int i = 0; i < n; i++) hogDetectionResult.add(null);

        //Resize the image.
        double scale = 64.0 / labelMinSize; //check statistics.txt to decide this scaling factor.
        int _width = (int)(figure.imageWidth * scale + 0.5), _height = (int)(figure.imageHeight * scale + 0.5);
        opencv_core.Size newSize = new opencv_core.Size(_width, _height);
        opencv_core.Mat imgScaled = new opencv_core.Mat(); resize(figure.imageGray, imgScaled, newSize);

        for (int i = 0; i < n; i++)
        {
            hog.setSVMDetector(new opencv_core.Mat(new FloatPointer(svmModels[i])));

            String panelLabelSet = labelSetsHOG[i];
//            double minSize = labelMinSize * scale;
//            double maxSize = labelMaxSize * scale;

            //Search on scaled image
            ArrayList<Panel> candidates1 = DetectMultiScale(imgScaled, panelLabelSet, false);
            //ArrayList<Panel> candidates2 = DetectMultiScale(imgeScaledInverted, panelLabelSet, true);

            ArrayList<Panel> candidates = new ArrayList<>();
            if (candidates1 != null) candidates.addAll(candidates1);
            //if (candidates2 != null) candidates.addAll(candidates2);

            if (candidates.size() > 0)
            {
                //Scale back to the original size, and save the result to hogDetectionResult
                ArrayList<Panel> segmentationResult = new ArrayList<>();
                for (int j = 0; j < candidates.size(); j++)
                {
                    Panel panel = candidates.get(j);
                    Rectangle rect = panel.labelRect;
                    Rectangle orig_rect = new Rectangle((int)(rect.x / scale + .5), (int)(rect.y / scale + .5), (int)(rect.width / scale + .5), (int)(rect.height / scale + .5));
                    panel.labelRect = orig_rect;

                    //Size checking. Ignore the size which is too small or too large
                    if (orig_rect.width > labelMaxSize || orig_rect.height > labelMaxSize) continue;
                    if (orig_rect.width < labelMinSize || orig_rect.height < labelMinSize) continue;

                    //Position check. Ignore the rect, when at least half of it is outside the original image.
                    //noticed that we have padded the figure image.
                    int centerX = orig_rect.x + orig_rect.width / 2;
                    int centerY = orig_rect.y + orig_rect.height / 2;
                    if (centerX <= Figure.padding || centerX >= figure.imageGray.cols() - Figure.padding) continue;
                    if (centerY <= Figure.padding || centerY >= figure.imageGray.rows() - Figure.padding) continue;

                    segmentationResult.add(panel);
                }
                candidates = RemoveOverlappedCandidates(segmentationResult);
                hogDetectionResult.set(i, candidates);
            }
        }
    }

    private ArrayList<Panel> DetectMultiScale(opencv_core.Mat img, String panelLabelSet, Boolean inverted)
    {
        ArrayList<Panel> candidates = new ArrayList<>();

        double hitThreshold = 0;			//Threshold for the distance between features and SVM classifying plane.
        opencv_core.Size winStride = new opencv_core.Size(8, 8); //Sliding window step, It must be a multiple of block stride
        opencv_core.Size padding = new opencv_core.Size(0, 0);	//Adds a certain amount of extra pixels on each side of the input image
        //Size padding = new Size(32, 32);	//Adds a certain amount of extra pixels on each side of the input image
        double scale0 = 1.05;			//Coefficient of the detection window increase
        int groupThreshold = 2;
        boolean useMeanShiftGrouping = false;

        opencv_core.RectVector rectVector = new opencv_core.RectVector();			DoublePointer dp = new DoublePointer();
        //hog.detectMultiScale(img, rectVector, dp);
        hog.detectMultiScale(img, rectVector, dp, hitThreshold, winStride, padding, scale0, groupThreshold, useMeanShiftGrouping);

        if (rectVector == null || rectVector.size() == 0) return null;

        double[] scores = new double[(int)rectVector.size()]; dp.get(scores);
        for (int k = 0; k < rectVector.size(); k++)
        {
            opencv_core.Rect labelRect = rectVector.get(k);

            Panel panel = new Panel();
            panel.labelRect = new Rectangle(labelRect.x(), labelRect.y(), labelRect.width(), labelRect.height());
            panel.panelLabel = panelLabelSet;
            //panel.labelInverted = inverted;
            panel.labelScore = scores[k];

            candidates.add(panel);
        }
        return candidates;
    }

    /**
     * The simplest method to merge label detection results saved in hogDetectionResult to panels (where the segmentation results are saved) <p>
     * This method simply combine all detected results
     */
    protected void mergeDetectedLabelsSimple()
    {
        figure.panels = new ArrayList<>(); //Reset
        if (hogDetectionResult == null) return;

        for (int i = 0; i < hogDetectionResult.size(); i++)
        {
            ArrayList<Panel> result = hogDetectionResult.get(i);
            if (result == null) continue;
            for (int j = 0; j < result.size(); j++)
                figure.panels.add(result.get(j));
        }
    }

    /**
     * The main entrance function to perform segmentation.
     * Call getResult* functions to retrieve result in different format.
     */
    void segment(opencv_core.Mat image)
    {
        figure = new Figure(image); //Common initializations for all segmentation method.

        HoGDetect();		//HoG Detection, detected patches are stored in hogDetectionResult

        //Merge all hogDetectionResult to segmentationResult
        mergeDetectedLabelsSimple();
    }


}

