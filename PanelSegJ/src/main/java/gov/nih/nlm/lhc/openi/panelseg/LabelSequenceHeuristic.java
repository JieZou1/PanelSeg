package gov.nih.nlm.lhc.openi.panelseg;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * For trying various postprocessing heuristics
 *
 * Created by jzou on 12/7/2016.
 */
public class LabelSequenceHeuristic
{
    private Figure figure;

    LabelSequenceHeuristic(Figure figure)
    {
        this.figure = figure;
    }

    void doHeuristic()
    {
        if (figure.panels.size() == 0) return;

        List<LabelBeamSearch.BeamItem> lastBeam = figure.beams.get(figure.beams.size()- 1);
        LabelBeamSearch.BeamItem bestItem = lastBeam.get(0); //For now, we use the optimal BeamItem only

        List<Panel> candidates = collectCandiatesFromBeam(bestItem);

        //Remove low prob candidates
        if (candidates != null && candidates.size()>0)
            candidates = removeFalseAlarmsByProb(candidates, 0.5);
        //Remove non-continuous label candidates
        if (candidates != null && candidates.size()>0)
            candidates = removeFalseAlarmsByLabels(candidates, 2);

        candidates = removeSinglePanelResult(candidates);

        //candidates = recoverBySearchingAlignment(candidates, figure);
        //candidates = removeNoAlignmentResult(candidates);

        // then save the result to figure.panels.
        figure.panels = candidates;

    }


    private  List<Panel> collectCandiatesFromBeam(LabelBeamSearch.BeamItem item)
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
        return panels;
    }

    private List<Panel> removeFalseAlarmsByProb(List<Panel> panels, double threshold)
    {
        List<Panel> candidates = new ArrayList<>();
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            if (panel.labelScore < threshold) continue;
            candidates.add(panel);
        }
        return candidates;
    }

    private List<Panel> removeFalseAlarmsByLabels(List<Panel> panels, int allowedInterval)
    {
        panels.sort(new PanelLabelAscending());

        List<Panel> candidates = new ArrayList<>();
        candidates.add(panels.get(0));
        int prevChar = Character.toLowerCase(panels.get(0).panelLabel.charAt(0));
        for (int i = 1; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            int currChar = Character.toLowerCase(panel.panelLabel.charAt(0));
            if (currChar - prevChar <= allowedInterval)
            {
                prevChar = currChar;
                candidates.add(panel);
            }
            else break;
        }
        return candidates;
    }

    private List<Panel> recoverBySearchingAlignment(List<Panel> panels, Figure figure)
    {
        if (panels.size() == 0) return panels;

        LabelSequenceClassify.SequenceType sequenceType = detectSequenceType(panels);
        if (sequenceType == LabelSequenceClassify.SequenceType.Digit) return panels; //We do not care for digits for now

        Panel.sortPanelsByNonNegProbs(panels); //Sort panels according to their non-neg probs

        //We search alignment on panels which have high prob to be labels only.
        List<Panel> candidates = new ArrayList<>();
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            if (panel.labelScore < 0.9) break;
            candidates.add(panel);
        }

        if (candidates.size() == 0) {return candidates;}

        //Split the candidates into alignment sets
        List<List<Panel>> sets = new ArrayList<>();
        while (candidates.size() != 0)
        {
            List<Panel> set = new ArrayList<>();
            List<Panel> left = new ArrayList<>();
            set.add(candidates.get(0));
            for (int i = 1; i < candidates.size(); i++)
            {
                Panel candidate = candidates.get(i);
                if (Panel.aligned(candidate, set)) set.add(candidate);
                else left.add(candidate);
            }
            sets.add(set);
            candidates = left;
        }

        //Sort all sets according their sizes (larger first)
        Collections.sort(sets, new Comparator<List>(){
            public int compare(List a1, List a2) {
                return a2.size() - a1.size(); // we want biggest to smallest
            }
        });

        //For now, we pick the longest set only
        candidates = sets.get(0);
        for (int i = 0; i < candidates.size(); i++) Panel.setLabelByNonNegProbs(candidates.get(i));

        boolean added = true;
        while (added)
        {
            added = false;

            //Find all aligned panels
            List<Panel> alignedPanels = new ArrayList<>();
            for (int i = 0; i < figure.panels.size(); i++)
            {
                Panel panel = figure.panels.get(i);

                boolean alreadyInCandidates = false;
                for (int j = 0; j < candidates.size(); j++)
                {
                    Panel candidate = candidates.get(j);
                    if (panel == candidate)
                    {
                        alreadyInCandidates = true;
                        break;
                    }
                }
                if (alreadyInCandidates) continue;

                for (int j = 0; j < candidates.size(); j++)
                {
                    Panel candidate = candidates.get(j);
                    if (Panel.aligned(panel, candidate))
                    {
                        alignedPanels.add(panel);
                        break;
                    }
                }
            }
            Panel.sortPanelsByNonNegProbs(alignedPanels);

            for (int i = 0; i < alignedPanels.size(); i++)
            {
                Panel panel = alignedPanels.get(i);
                if (panel.labelScore < 0.1) break;

                Panel.setLabelByNonNegProbs(panel);

                //Has to be the same SequenceType
                char ch = panel.panelLabel.charAt(0);
                if (!PanelSeg.isCaseSame(ch))
                {
                    if (Character.isLowerCase(ch) && sequenceType != LabelSequenceClassify.SequenceType.Lower) continue;
                    if (Character.isUpperCase(ch) && sequenceType != LabelSequenceClassify.SequenceType.Upper) continue;
                }

                if (!LabelSequenceClassify.noDuplicateLabels(panel, candidates)) continue;
                if (!LabelSequenceClassify.noOverlappingRect(panel, candidates)) continue;

                candidates.add(panel);
                added = true;
            }
        }
        return candidates;
    }

    private LabelSequenceClassify.SequenceType detectSequenceType(List<Panel> panels)
    {
        int countDigit = 0, countUpper = 0, countLower = 0;
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            char ch = panel.panelLabel.charAt(0);
            if (Character.isDigit(ch)) countDigit++;
            else
            {
                if (PanelSeg.isCaseSame(ch))
                {
                    countUpper++; countLower++;
                }
                else
                {
                    if (Character.isLowerCase(ch)) countLower++;
                    else countUpper++;
                }
            }

        }

        if (countDigit > countUpper && countDigit > countLower) return LabelSequenceClassify.SequenceType.Digit;
        if (countLower > countUpper && countLower > countDigit) return LabelSequenceClassify.SequenceType.Lower;
        if (countUpper > countLower && countUpper > countDigit) return LabelSequenceClassify.SequenceType.Upper;
        return LabelSequenceClassify.SequenceType.Upper;
    }

    private List<Panel> removeNoAlignmentResult(List<Panel> panels)
    {
        List<Panel> candidates = new ArrayList<>();

        if (panels.size() <= 1) return candidates;

        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel1 = panels.get(i);
            for (int j = i + 1; j < panels.size(); j++)
            {
                Panel panel2 = panels.get(j);
                if (Panel.aligned(panel1, panel2)) return panels;
            }
        }

        return candidates;
    }

    private List<Panel> removeSinglePanelResult(List<Panel> panels)
    {
        List<Panel> candidates = new ArrayList<>();

        if (panels.size() <= 1) return candidates;

        return panels;
    }

    /**
     * To try recover the case where the labels are horizontally aligned and left-to-right
     */
    private void recoverHorizontalLeftRightGrid(List<Panel> panels)
    {
    }
}
