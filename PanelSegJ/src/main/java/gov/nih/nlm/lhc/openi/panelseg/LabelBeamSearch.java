package gov.nih.nlm.lhc.openi.panelseg;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import scala.Char;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Label sequence finding using Beam Search
 * Refactored from PanelSegLabelRegHogSvmBeam (which is obsolete) to make class hierarchy simpler.
 *
 * Created by jzou on 11/8/2016.
 */
final class LabelBeamSearch
{
    private Figure figure;

    LabelBeamSearch(Figure figure)
    {
        this.figure = figure;
    }

    void search()
    {
        List<Panel> panels = figure.panels;
        if (panels.size() == 0) return;

        Panel.sortPanelsByNonNegProbs(panels); //Sort panels according to their non-neg probs

        //Initialize the beam
        int n = panels.size(), beamLength = 100;
        List<List<BeamItem>> beams = new ArrayList<>();

        for (int i = 0; i < n; i++)
        {
            List<BeamItem> beam = new ArrayList<>();
            Panel panel = panels.get(i);

            //construct beams for this position
            for (int j = 0; j < panel.labelProbs.length; j++)
            {
                double p1 = panel.labelProbs[j];
                //If p1 (Patch classification post-prob) is too small, we don't consider it.
                if (j != PanelSeg.labelChars.length && p1 < 0.02) continue;

                if (i == 0)
                {
                    BeamItem item = new BeamItem();
                    item.labelIndexes.add(j);  item.p1 = Math.log(p1);

                    if (!isValidLabelSequence(item)) continue;

                    updateScore(item);
                    beam.add(item);
                }
                else
                {
                    List<BeamItem> prevBeam = beams.get(i - 1);
                    for (int k = 0; k < prevBeam.size(); k++)
                    {
                        BeamItem prevItem = prevBeam.get(k);

                        BeamItem item = new BeamItem();
                        item.labelIndexes.addAll(prevItem.labelIndexes);
                        item.labelIndexes.add(j);
                        item.p1 = prevItem.p1 + Math.log(p1);

                        if (!isValidLabelSequence(item)) continue;

                        sequenceClassify(item);
                        updateScore(item);
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
        List<LabelBeamSearch.BeamItem> lastBeam = beams.get(beams.size()- 1);
        //lastBeam = updateLastBeam(lastBeam);

        //Sort and then find the optimal sequence, and update the panels
        lastBeam.sort(new LabelBeamSearch.ScoreDescending());

        figure.beams = beams;
    }

    /**
     * Check the label sequence. If it is an illegal sequence, return false; otherwise return true.
     * @param item
     * @return
     */
    private boolean isValidLabelSequence(BeamItem item)
    {
        if (!LabelSequenceClassify.noDuplicateLabels(item)) return false;
        if (!LabelSequenceClassify.sameCaseLabels(item)) return false;
        if (!LabelSequenceClassify.noOverlappingRect(item, figure)) return false;

        return true;
    }

    /**
     * Classify sequence, assign item.p2 and item.score
     */
    private void sequenceClassify(BeamItem item)
    {
        List<Panel> labelPanels = new ArrayList<>();
        for (int i = 0; i < item.labelIndexes.size(); i++)
        {
            int index = item.labelIndexes.get(i);
            if (index == PanelSeg.labelChars.length) continue; //classified as neg,

            Panel panel = figure.panels.get(i);
            panel.panelLabel = "" + PanelSeg.labelChars[index];
            labelPanels.add(panel);
        }
        labelPanels.sort(new PanelLabelAscending());
        Panel[] panels = labelPanels.toArray(new Panel[labelPanels.size()]);

        if (LabelSequenceClassify.svmModels[panels.length] != null)
        {
            int order = panels.length;
            //Extract and normalize the features
            float[] feature = LabelSequenceClassify.featureExtraction(figure.imageOriginal, panels);
            for (int i = 0; i < feature.length; i++)
                feature[i] = (feature[i] - LabelSequenceClassify.mins[order][i]) / LabelSequenceClassify.ranges[order][i];

            svm_node[] svmNode = LibSvmEx.float2SvmNode(feature);

            svm_model svmModel = LabelSequenceClassify.svmModels[panels.length];
            double[] svmProbs = new double[LibSvmEx.getNrClass(svmModel)];
	        /*double label = */svm.svm_predict_probability(svmModel, svmNode, svmProbs);
	        item.p2 = Math.log(svmProbs[0]); //positive sequence prob;
        }
    }

    /**
     * UPdate item.score based on item.p1 and item.p2
     * @param item
     */
    private void updateScore (BeamItem item)
    {
        if (item.p2 > 0.0)
            item.score = item.p1 / item.labelIndexes.size();
        else
            item.score = (item.p1 + item.p2) / (item.labelIndexes.size() + 1);
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
            if (currChar - prevChar > 1) return false; //No missing more than 1
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


    class BeamItem
    {
        double score;
        double p1;  //log prob of label sequence given the patches, from patch classification
        double p2;  //log prob of label sequence given other info (bounding box position, size, etc.), from sequence classifier
        ArrayList<Integer> labelIndexes; //The panel label-index sequence up to this BeamItem.

        BeamItem()
        {
            labelIndexes = new ArrayList<>();
            p1 = p2 = Double.POSITIVE_INFINITY; //p1, p2 should be in the range (-inf, 0], set initial value to INF to indicate the value is not set.
        }
    }

    /**
     * Comparator for sorting Panels in reverse order of labelScore.
     * @author Jie Zou
     */
    class ScoreDescending implements Comparator<BeamItem>
    {
        public int compare(LabelBeamSearch.BeamItem o1, LabelBeamSearch.BeamItem o2)
        {
            double diff = o2.score - o1.score;
            if (diff > 0) return 1;
            else if (diff == 0) return 0;
            else return -1;
        }
    }

}
