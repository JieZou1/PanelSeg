package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;

import java.util.ArrayList;

/**
 * HOG + SVM method for Label Recognition
 * HOG for label detection, SVM for label classification and simple threshold method for make the final decision
 *
 * Created by jzou on 9/21/2016.
 */
public class PanelSegLabelRegHogSvmThresholding extends PanelSegLabelRegHogSvm
{
    public PanelSegLabelRegHogSvmThresholding()
    {
        super();
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
        SvmClassification();			//SVM classification of each detected patch in figure.panels.
        MergeRecognitionLabelsSimple();

        //additional steps for this method.
        if (figure.panels.size() == 0) return;

        //Use simple thresholding to pick the final result
        double threshold = 0.98;

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
