package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_objdetect.HOGDescriptor;

import java.lang.reflect.Field;

/**
 * Treat Label Recognition as Object Detection using HoG method
 *
 * Created by jzou on 8/31/2016.
 */
public class PanelSegLabelRegHog extends PanelSegLabelReg
{
    static String[] labelSetsHOG = {"A", "ad", "BDEFPpR", "bhKk", "CceGOoQ", "fIiJjLlrTt", "gqSs", "HMmNn" };

    //The HoG parameters used in both training and testing
    static protected opencv_core.Size winSize_64 = new opencv_core.Size(64, 64);
    //static private Size winSize_32 = new Size(32, 32); //The size of the training label patches
    static private opencv_core.Size blockSize = new opencv_core.Size(16, 16);
    static private opencv_core.Size blockStride = new opencv_core.Size(8, 8);
    static private opencv_core.Size cellSize = new opencv_core.Size(8, 8);
    static private int nbins = 9;
//  static private int derivAperture = 1;
//  static private double winSigma = -1;
//  static private double L2HysThreshold = 0.2;
//  static private boolean gammaCorrection = true;
//  static private int nLevels = 64;

    //The HoG parameters used in testing only.
    static private double hitThreshold = 0;			//Threshold for the distance between features and SVM classifying plane.
    static private opencv_core.Size winStride = new opencv_core.Size(8, 8); //Sliding window step, It must be a multiple of block stride
    static private opencv_core.Size padding = new opencv_core.Size(0, 0);	//Adds a certain amount of extra pixels on each side of the input image
    //static private Size padding = new Size(32, 32);	//Adds a certain amount of extra pixels on each side of the input image
    static private double scale0 = 1.05;			//Coefficient of the detection window increase
    static private int groupThreshold = 2;
    static private boolean useMeanShiftGrouping = false;

    static private double minimumLabelSize = 12.0;	//We assume the smallest label patch is 12x12.

    private HOGDescriptor hog;
    private float[][] svmModels;

    public PanelSegLabelRegHog()
    {
        int n = labelSetsHOG.length;		svmModels = new float[n][];
        for (int i = 0; i < n; i++)
        {
            String classString = "gov.nih.nlm.iti.figure.PanelSegLabelRegHoGModel_" + labelSetsHOG[i];
            try {
                Class<?> cls = Class.forName(classString);
                Field field = cls.getField("svmModel");
                svmModels[i] = (float[])field.get(null);
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }

        hog = new HOGDescriptor(winSize_64, blockSize, blockStride, cellSize, nbins);
        //hog = new HOGDescriptor(winSize_32, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, _histogramNormType, _L2HysThreshold, gammaCorrection, nlevels, _signedGradient)
    }

}
