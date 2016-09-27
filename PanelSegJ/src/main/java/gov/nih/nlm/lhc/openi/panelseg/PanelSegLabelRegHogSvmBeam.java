package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;

import java.awt.*;
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

        //Sort the candidates according their max prob, before Beam Search,
        //Check whether in this case we have better chance to not miss the correct sequence during the beam search.
        //sortCandidates();

        beamSearch();
    }

    /**
     * Sort the candidates according their max prob
     */
    private void sortCandidates()
    {
        //set label and score according to the max of labelProbs, computed by SVM
        ArrayList<Panel> candidates = new ArrayList<>();
        for (int j = 0; j < figure.panels.size(); j++)
        {
            Panel panel = figure.panels.get(j);
            int maxIndex = AlgMiscEx.findMaxIndex(panel.labelProbs);
            panel.labelScore = panel.labelProbs[maxIndex];
            //panel.panelLabel = "" + labelChars[maxIndex];
            candidates.add(panel);
        }
        //candidates.sort(new LabelScoreDescending());
        //candidates.sort(new LabelScoreAscending());
        figure.panels = candidates;
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
                //If p1 (Classification post-prob) is too small, we don't consider it.
                if (j != PanelSeg.labelChars.length && p1 < 0.02) continue;

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

                        if (checkLabelSequence(item)) beam.add(item);
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
        lastBeam = updateLastBeam(lastBeam);

        //Sort and then find the optimal sequence, and update the panels
        lastBeam.sort(new ScoreDescending());
        BeamItem bestItem = lastBeam.get(0);

        //Finally check to remove panels which looks obviously wrong, to increase precision
        // then save the result to figure.panels.
        figure.panels = finalCheck(bestItem);
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
        if (!noOverlappingRect(item)) return false;

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

    private boolean noOverlappingRect(BeamItem item)
    {
        //Collect all label rects.
        List<Rectangle> rectangles = new ArrayList<>();
        for (int i = 0; i < item.labelIndexes.size(); i++)
        {
            int index = item.labelIndexes.get(i);
            if (index == PanelSeg.labelChars.length) continue; //classified as negative don't care.

            Panel panel = figure.panels.get(i);
            rectangles.add(panel.labelRect);
        }

        for (int i = 0; i < rectangles.size(); i++)
        {
            Rectangle r1 = rectangles.get(i);
            for (int j = i+ 1; j < rectangles.size(); j++)
            {
                Rectangle r2 = rectangles.get(j);
                Rectangle intersect = r1.intersection(r2);
                if (!intersect.isEmpty()) return false;
            }
        }

        return true;
    }

    /**
     * Check the last beam, updates its joint prob and remove illegal beam items.
     * @param beam
     * @return
     */
    private List<BeamItem> updateLastBeam(List<BeamItem> beam)
    {
        List<BeamItem> candidates = new ArrayList<>();
        for (int i = 0; i < beam.size(); i++)
        {
            BeamItem item = beam.get(i);
            if (!updateItem(item)) continue;

            candidates.add(item);
        }
        if (candidates.size() == 0) return beam; //no one is qualified, we keep the original
        return candidates;
    }

    /**
     * Update the BeamItem prob, for other evidences, such as length K, consequence no missing,
     * @param item
     */
    private boolean updateItem(BeamItem item)
    {
        //Collect all panels
        List<Panel> panels = new ArrayList<>();
        for (int i = 0; i < item.labelIndexes.size(); i++)
        {
            int labelIndex = item.labelIndexes.get(i);
            if (labelIndex == PanelSeg.labelChars.length) continue; //Classified as a negative sample.

            Panel panel = figure.panels.get(i);
            panel.panelLabel = "" + PanelSeg.labelChars[labelIndex];
            panel.labelScore = panel.labelProbs[labelIndex];
            panels.add(panel);
        }
        panels.sort(new PanelLabelAscending());

        //Check label sequence consecutive. It can not miss more than 2
        if (panels.size() == 0) return true; //No labels, which is possible
        if (panels.size() <= 1) return false; //Only one label, not good.
        int prevChar = Character.toLowerCase(panels.get(0).panelLabel.charAt(0));
        for (int i = 1; i < panels.size(); i++)
        {
            int currChar = Character.toLowerCase(panels.get(i).panelLabel.charAt(0));
            if (currChar - prevChar > 1) return false;
            prevChar = currChar;
        }

        return true;
    }

    private List<Panel> finalCheck(BeamItem item)
    {
        //Collect all panels
        List<Panel> panels = new ArrayList<>();
        for (int i = 0; i < item.labelIndexes.size(); i++)
        {
            int labelIndex = item.labelIndexes.get(i);
            if (labelIndex == PanelSeg.labelChars.length) continue; //Classified as a negative sample.

            Panel panel = figure.panels.get(i);
            panel.panelLabel = "" + PanelSeg.labelChars[labelIndex];
            panel.labelScore = panel.labelProbs[labelIndex];
            panels.add(panel);
        }
        panels.sort(new PanelLabelAscending());

        if (panels.size() <= 1) return panels;

        List<Panel> candidates = new ArrayList<>();
        candidates.add(panels.get(0));
        int prevChar = Character.toLowerCase(panels.get(0).panelLabel.charAt(0));
        for (int i = 1; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            int currChar = Character.toLowerCase(panel.panelLabel.charAt(0));
            if (currChar - prevChar <= 1)
            {
                prevChar = currChar;
                candidates.add(panel);
            }
            else break;
        }
        return candidates;
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


