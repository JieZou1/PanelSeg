package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;

import java.util.ArrayList;
import java.util.List;

/**
 * HOG + SVM method for Label Recognition
 * HOG for label detection, SVM for label classification and some rules to pick up the final result
 *
 * Created by jzou on 9/21/2016.
 */
public class PanelSegLabelRegHogSvmBeam extends  PanelSegLabelRegHogSvm
{
    public PanelSegLabelRegHogSvmBeam()
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

        //additional steps for this method.
        if (figure.panels.size() == 0) return;

    }

    private List<Panel> upperLetterCandidates;
    private List<Panel> lowerLetterCandidates;
    private List<Panel> digitCandidates;


    /**
     * We assume the labels should all the Upper-Case, or Lower-Case, or Digits
     */
    private void collectSequences()
    {
        upperLetterCandidates = new ArrayList<>();
        lowerLetterCandidates = new ArrayList<>();
        digitCandidates = new ArrayList<>();

        for (Panel panel: figure.panels)
        {
            String label = panel.panelLabel;
            Character ch = label.charAt(0); //For now, we could detect single char label only

            if (Character.isDigit(ch))
                digitCandidates.add(panel);
            else if (ch == 'c' || ch == 'C' ||
                     ch == 'i' || ch == 'I' ||
                     ch == 'j' || ch == 'J' ||
                     ch == 'k' || ch == 'K'
                    )
            {
                upperLetterCandidates.add(panel);
                lowerLetterCandidates.add(panel);
            }

        }
    }
}


