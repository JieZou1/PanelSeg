package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;

import java.lang.reflect.Array;
import java.nio.charset.CharacterCodingException;
import java.util.ArrayList;
import java.util.Comparator;
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

        beamSearch();
    }

    private void beamSearch()
    {
        //Initialize the beam
        int n = figure.panels.size(), beamLength = 100;
        List<List<BeamItem>> beams = new ArrayList<>();

        for (int i = 0; i < n; i++)
        {
            List<BeamItem> beam = new ArrayList<>();
            Panel panel = figure.panels.get(i);

            //construct beams for this position
            for (int j = 0; j < panel.labelProbs.length; j++)
            {
                double p1 = panel.labelProbs[j];
                if (i == 0)
                {
                    BeamItem item = new BeamItem();
                    item.p1 = Math.log(p1);
                    item.labelIndexes.add(j);

                    if (checkLabelSequence(item)) beam.add(item);
                }
                else
                {
                    List<BeamItem> prevBeam = beams.get(i - 1);
                    for (int k = 0; k < prevBeam.size(); k++)
                    {
                        BeamItem prevItem = prevBeam.get(k);

                        BeamItem item = new BeamItem();
                        item.p1 = prevItem.p1 + Math.log(p1);
                        item.labelIndexes.addAll(prevItem.labelIndexes);
                        item.labelIndexes.add(j);

                        if (checkLabelSequence(item))
                            beam.add(item);
                    }
                }
            }

            //sort beam items in this position, and keep only top beamLength items.
            beam.sort(new ScoreDescending());
            if (beam.size() > beamLength) beam = beam.subList(0, beamLength);
            beams.add(beam);
        }

        //Update prob for the last beam
        List<BeamItem> lastBeam = beams.get(beams.size()- 1);
        for (int i = 0; i < lastBeam.size(); i++)
        {
            BeamItem item = lastBeam.get(i);
            updateItem(item);
        }
        lastBeam.sort(new ScoreDescending());

        //Find the optimal option, and update the panels
        BeamItem bestItem = lastBeam.get(0);
        ArrayList<Panel> candidates = new ArrayList<>();
        for (int i = 0; i < figure.panels.size(); i++)
        {
            int labelIndex = bestItem.labelIndexes.get(i);
            if (labelIndex == PanelSeg.labelChars.length) continue; //Classified as a negative sample.

            Panel panel = figure.panels.get(i);
            panel.panelLabel = "" + PanelSeg.labelChars[labelIndex];
            panel.labelScore = panel.labelProbs[labelIndex];
            candidates.add(panel);
        }
        figure.panels = candidates;
    }

    /**
     * Check the label sequence, update item.score by calculating item.p1, item.p2 and item.p3, and item.score
     * if it is an illegal sequence, return false; otherwise return true.
     * @param item
     * @return
     */
    private boolean checkLabelSequence(BeamItem item)
    {
        if (!noDuplicateLabels(item)) return false;
        if (!sameCaseLabels(item)) return false;

        item.score = item.p1 + item.p2 + item.p3;

        return true;
    }

    private boolean noDuplicateLabels(BeamItem item)
    {
        List<Character> charsLower = new ArrayList<>();
        for (int i = 0; i < item.labelIndexes.size(); i++)
        {
            int labelIndex = item.labelIndexes.get(i);
            if (labelIndex == PanelSeg.labelChars.length) continue; //None label patch, we allow duplicates of course.

            char ch = PanelSeg.labelChars[labelIndex];
            char chLower = Character.toLowerCase(ch);

            if (charsLower.indexOf(chLower) != -1) return false;

            charsLower.add(chLower);
        }
        return true;
    }

    private boolean sameCaseLabels(BeamItem item)
    {
        SequenceType typeSeq = SequenceType.Unknown;
        for (int i = 0; i < item.labelIndexes.size(); i++)
        {
            int labelIndex = item.labelIndexes.get(i);
            if (labelIndex == PanelSeg.labelChars.length) continue; //None label patch

            char ch = PanelSeg.labelChars[labelIndex];

            SequenceType type;
            if (Character.isDigit(ch))          type = SequenceType.Digit;
            else if (Character.isLowerCase(ch)) type = SequenceType.Lower;
            else                                type = SequenceType.Upper;

            if (type == SequenceType.Digit)
            {
                if (typeSeq == SequenceType.Unknown) typeSeq = type;
                else
                {
                    if (type != typeSeq) return false;
                }
            }
            else if (type == SequenceType.Upper)
            {
                if (typeSeq == SequenceType.Digit) return false;
                if (PanelSeg.isCaseSame(ch)) continue;

                if (typeSeq == SequenceType.Unknown) typeSeq = type;
                else
                {
                    if (type != typeSeq) return false;
                }
            }
            else if (type == SequenceType.Lower)
            {
                if (typeSeq == SequenceType.Digit) return false;
                if (PanelSeg.isCaseSame(ch)) continue;

                if (typeSeq == SequenceType.Unknown) typeSeq = type;
                else
                {
                    if (type != typeSeq) return false;
                }
            }
        }
        return true;
    }

    /**
     * Update the BeamItem prob, for other evidences, length K, consequence no missing,
     * @param item
     */
    private void updateItem(BeamItem item)
    {

    }

    enum SequenceType {Unknown, Upper, Lower, Digit}

    class BeamItem
    {
        double score;
        double p1;  //prob of label sequence given the patches, from patch classification (SVM)
        double p2;  //prob of label sequence (same case, no duplicates, etc.)
        double p3;  //prob of label sequence given other info (bounding box position, size, etc.)
        ArrayList<Integer> labelIndexes; //The panel label-index sequence up to this BeamItem.

        BeamItem()
        {
            labelIndexes = new ArrayList<>();
            p1 = p2 = p3 = 0;
        }
    }

    /**
     * Comparator for sorting Panels in reverse order of labelScore.
     * @author Jie Zou
     */
    class ScoreDescending implements Comparator<BeamItem>
    {
        public int compare(BeamItem o1, BeamItem o2)
        {
            double diff = o2.score - o1.score;
            if (diff > 0) return 1;
            else if (diff == 0) return 0;
            else return -1;
        }
    }

}


