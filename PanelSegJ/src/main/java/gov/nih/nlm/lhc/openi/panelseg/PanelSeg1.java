package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;

import java.awt.*;
import java.util.*;
import java.util.List;

import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_highgui.waitKey;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

/**
 * The complete Panel Segmentation algorithm (Panel Label Recognition + Panel Splitting)
 *
 * The basic idea is:
 * 1. Use Uniform Bands, Straight Line Edges, and Connected Components to propose candidate splitting regions.
 * 2. Detect and recognize Panel Labels (compute posterior probabilities)
 * 3. Use Beam Search to find best matches.
 *      One label matches to one region and
 *      maximize the joint label posterior probabilities
 *
 * Created by jzou on 5/8/2017.
 */
public class PanelSeg1
{
    Figure figure;

    List<Panel> panels;         //The panels
    List<List<Panel>> labelSets; //The list of labels assigned to each of the panels
    private List<Panel> panelsT;         //Temp panels
    private List<List<Panel>> labelSetsT; //Temp list of labels assigned to each of the panels

    static void initialize(Properties properties) throws Exception
    {
        LabelDetectHog.initialize(properties);
        LabelClassifyLeNet5.initialize(properties);
        LabelClassifyHogSvm.initialize(properties);
        LabelSequenceClassify.initialize(properties);

        PanelSplitStructuredEdge.initialize();
    }

    PanelSeg1(Figure figure) {this.figure = figure; }

    void segment() throws Exception
    {
        //Split based on Uniform Bands
        figure.panelsUniformBand = splitUniformBands();

        //Label Recognize using HoG + SVM + Beam Search
        //figure.panelsLabelHoGSvm = labelRegHogSvm();
        figure.panelsLabelHoGSvmBeam = labelRegHogSvmBeam();

        if (figure.panelsLabelHoGSvmBeam.size() == 0)
        {
            if (figure.panelsUniformBand.size() == 1)
            {
                //No labelSets are detected, and there is only one panel.
                //This is reasonable to be a single panel figure without labelSets
                figure.panels = figure.panelsUniformBand;
            }
            else
            {
                //Check panelsUniformBand
                Boolean valid = true;
                for (Panel panel : figure.panelsUniformBand)
                {
                    if (panel.panelRect.width < 100 && panel.panelRect.height < 100)
                        valid = false;
                }
                if (!valid)
                {
                    Panel panel = new Panel();
                    panel.panelRect = new Rectangle(50, 50, figure.imageOriginalWidth, figure.imageOriginalHeight);
                    figure.panels = new ArrayList<>();
                    figure.panels.add(panel);
                }
                else
                    figure.panels = figure.panelsUniformBand; //For now, we accept uniform-band result too.
            }
        }
        else
        {   //Merge/split panels with labelSets to generate final result
            this.panels = new ArrayList<>();
            this.panels.addAll(figure.panelsUniformBand);

            //Merge split-panels and recognized-panel-labelSets
            this.labelSets = matchPanels(panels, figure.panelsLabelHoGSvmBeam); //each panel in panels corresponds to an element in labelSets

            //check matched result. If valid (one-to-one matching is found), conclude the segmentation)
            if (isOne2OneMatch(panels, labelSets))
            {
                //Labels and Panels are one-to-one matched, we conclude the segmentation by setting figure.panels.
                figure.panels = setFigurePanels(panels, labelSets);
            }
            else
            {
                //Reach Here: panels and labelSets are merged and saved in this.panels and this.labelSets
                //1. Handle panels with matching labelSets, remove or further splitting
                //2. Handle panels without matching labelSets, merge to the closet one or adding a label
                structuredEdgeAnalysis();
            }
        }

        //displayResults();
    }

    List<Panel> splitUniformBands()
    {
        Figure figure = new Figure(this.figure.imageOriginal);
        PanelSplitJaylene splitJaylene = new PanelSplitJaylene(figure);
        splitJaylene.split();
        return figure.getSegResultWithPadding();
    }

    List<Panel> labelRegHogSvm() throws Exception
    {
        Figure figure = new Figure(this.figure.imageOriginal);
        LabelDetectHog detectHog = new LabelDetectHog(figure);
        detectHog.hoGDetect();        //HoG Detection, detected patches are stored in hogDetectionResult
        detectHog.mergeDetectedLabelsSimple();  //Merge all hogDetectionResult to panels

        LabelClassifyLeNet5 classifyLeNet5 = new LabelClassifyLeNet5(figure);
        classifyLeNet5.LeNet5Classification();    //LeNet5 classification of each detected patch in figure.panels.

        //Do label classification with HoG-SVM
        LabelClassifyHogSvm classifySvm = new LabelClassifyHogSvm(figure);
        classifySvm.svmClassificationWithLeNet5();

        return figure.getSegResultWithPadding();
    }

    List<Panel> labelRegHogSvmBeam() throws Exception
    {
        Figure figure = new Figure(this.figure.imageOriginal);

        LabelDetectHog detectHog = new LabelDetectHog(figure);
        detectHog.hoGDetect();        //HoG Detection, detected patches are stored in hogDetectionResult
        detectHog.mergeDetectedLabelsSimple();  //Merge all hogDetectionResult to panels

        LabelClassifyLeNet5 classifyLeNet5 = new LabelClassifyLeNet5(figure);
        classifyLeNet5.LeNet5Classification();    //LeNet5 classification of each detected patch in figure.panels.
        classifyLeNet5.removeFalseAlarms(0.02);

        //Do label classification with HoG-SVM
        LabelClassifyHogSvm classifySvm = new LabelClassifyHogSvm(figure);
        classifySvm.svmClassificationWithLeNet5();
        //classifySvm.mergeRecognitionLabelsSimple();

        LabelBeamSearch beamSearch = new LabelBeamSearch(figure);
        beamSearch.search();

        LabelSequenceHeuristic heuristic = new LabelSequenceHeuristic(figure);
        heuristic.doHeuristic();

        return figure.getSegResultWithPadding();
    }

    List<List<Panel>> matchPanels(List<Panel> panels, List<Panel> labels)
    {
        List<List<Panel>> labelSets = new ArrayList<>();
        for (int i = 0; i < panels.size(); i++)  labelSets.add(new ArrayList<>());

        //Assign labelSets to the closet panels
        for (int i = 0; i < labels.size(); i++)
        {
            Panel label = labels.get(i);

            //find max overlapping
            int maxIndex = AlgMiscEx.maxOverlappingPanel(label.labelRect, panels);
            if (maxIndex != -1)
            {
                labelSets.get(maxIndex).add(label);
                continue;
            }

            //No overlapping, we find the closest panel
            int minIndex = AlgMiscEx.closestPanel(label.labelRect, panels);
            labelSets.get(minIndex).add(label);
        }
        return labelSets;
    }

    boolean isOne2OneMatch(List<Panel> panels, List<List<Panel>> labelSets)
    {
        for (int i = 0; i < panels.size(); i++)
        {
            List<Panel> labelSet = labelSets.get(i);
            if (labelSet == null) return false;
            if (labelSet.size() != 1) return false;
        }
        return true;
    }

    boolean isValidSplit(List<Panel> panels, List<List<Panel>> labelSets)
    {
        for (int i = 0; i < panels.size(); i++)
        {
            List<Panel> labelSet = labelSets.get(i);
            if (labelSet == null || labelSet.size() == 0) return false;
        }
        return true;
    }

    /**
     */
    List<Panel> setFigurePanels(List<Panel> panels, List<List<Panel>> labelSets)
    {
        List<Panel> panelsNew = new ArrayList<>();
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            List<Panel> labelSet = labelSets.get(i);
            if (labelSet!=null && labelSet.size() >= 1)
            {
                Panel label = labelSet.get(0);
                label.panelRect = panel.panelRect;
                panelsNew.add(label);
            }
            else
                panelsNew.add(panel);
        }
        return panelsNew;
    }

    void structuredEdgeAnalysis()
    {
        PanelSplitStructuredEdge splitStructuredEdge = new PanelSplitStructuredEdge(figure);
        splitStructuredEdge.detectEdges();
        splitStructuredEdge.ccAnalysis();

        //Process panels which has matching labels.
        // We either remove the extra labels, or split into more panels
        panelsT = new ArrayList<>(); labelSetsT = new ArrayList<>();
        for (int i = 0; i < this.panels.size(); i++)
        {
            Panel panel = this.panels.get(i);   List<Panel> labelSet = this.labelSets.get(i);

            if (labelSet.size() == 0 || labelSet.size() == 1) {
                panelsT.add(panel);
                labelSetsT.add(labelSet);
                continue;
            }

            //Recursively try splitting the panel if more than one labels are linked to this panel
            //If the split panel contains no labels, we reject this split.
            Queue<Panel> toSplitPanels = new LinkedList<>();            toSplitPanels.add(panel);
            Queue<List<Panel>> toSplitLabelSets = new LinkedList<>();   toSplitLabelSets.add(labelSet);
            while (toSplitPanels.size() > 0)
            {
                Panel toSplitPanel = toSplitPanels.poll();  List<Panel> toSplitLabelSet = toSplitLabelSets.poll();

                List<Panel> panelsSplit = splitStructuredEdge.splitByBandAndLine(toSplitPanel);
                if (panelsSplit.size() != 0)
                {   //Able to split by structured edges
                    if (panelsSplit.size() == toSplitLabelSet.size())
                    {   //The panels happens to be the same as label set, we simply assigns the labels to each of the panels
                        int horOverlap, verOverlap;
                        {
                            panelsSplit.sort(new PanelRectLeftAscending());
                            Rectangle rect0 = panelsSplit.get(0).panelRect, rect1 = panelsSplit.get(1).panelRect;
                            horOverlap = (rect1.x + rect1.width - rect0.x) - (rect0.width + rect1.width);
                        }
                        {
                            panelsSplit.sort(new PanelRectTopAscending());
                            Rectangle rect0 = panelsSplit.get(0).panelRect, rect1 = panelsSplit.get(1).panelRect;
                            verOverlap = (rect1.y + rect1.height - rect0.y) - (rect0.height + rect1.height);
                        }
                        if (horOverlap > verOverlap)
                        {
                            panelsSplit.sort(new PanelRectLeftAscending());
                            toSplitLabelSet.sort(new LabelRectLeftAscending());
                        }
                        else
                        {
                            panelsSplit.sort(new PanelRectTopAscending());
                            toSplitLabelSet.sort(new LabelRectTopAscending());
                        }

                        for (int k = 0; k < panelsSplit.size(); k++)
                        {
                            Panel panelSplit = panelsSplit.get(k);
                            List<Panel> labelSetSplit = new ArrayList<>();
                            labelSetSplit.add(toSplitLabelSet.get(k));
                            panelsT.add(panelSplit); labelSetsT.add(labelSetSplit);
                        }
                        continue;
                    }
                    else
                    {
                        List<List<Panel>> labelSetsSplit = matchPanels(panelsSplit, toSplitLabelSet);
                        if (isValidSplit(panelsSplit, labelSetsSplit))
                        {   //Valid split means no newly created panels without labels
                            for (int k = 0; k < panelsSplit.size(); k++)
                            {
                                Panel panelSplit = panelsSplit.get(k);
                                List<Panel> labelSetSplit = labelSetsSplit.get(k);
                                if (labelSetSplit.size() == 1)
                                {
                                    panelsT.add(panelSplit); labelSetsT.add(labelSetSplit);
                                }
                                else
                                {
                                    toSplitPanels.add(panelSplit); toSplitLabelSets.add(labelSetSplit);
                                }
                            }
                            continue;
                        }
                    }
                }

                //Reach Here: Not able to split by BandAndLine analysis of structured edges
                panelsSplit = splitStructuredEdge.splitByCCAnalysis(toSplitPanel, toSplitLabelSet);
                if (panelsSplit.size() > 0)
                {
                    panelsT.addAll(panelsSplit);
                    for (Panel p : panelsSplit)
                    {
                        List<Panel> set = new ArrayList<>(); set.add(p);
                        labelSetsT.add(set);
                    }
                    continue;
                }

                //Reach Here: Not able to split by both BandAndLine and CC analysis of structured edges
                //We simple keep the smaller label for now.
                List<Panel> smallestLabel = new  ArrayList<>();
                smallestLabel.add(smallestLabel(toSplitLabelSet));
                panelsT.add(toSplitPanel); labelSetsT.add(smallestLabel);
            }
        }

        //complete analysis, save the result
        panels = panelsT; labelSets = labelSetsT;

        //Check again
        if (isOne2OneMatch(panels, labelSets))
        {
            //Labels and Panels are one-to-one matched, we conclude the segmentation by setting figure.panels.
            figure.panels = setFigurePanels(panels, labelSets);
            return;
        }

        //ToDo: Process panels which has no matching labels.
        //We either merge it into the existing panel or add labels
        panelsT = setFigurePanels(panels, labelSets);

        if (tryRowFirst(panelsT))
        {
            figure.panels = panelsT;
            return;
        }

        //complete analysis, save the result
        figure.panels = panelsT;
    }

    /**
     * For the case when panels are aligned row by row.
     * @param panels
     */
    private boolean tryRowFirst(List<Panel> panels)
    {
        panels.sort(new PanelRectRowFirst());

        //Also find all panels, which has been assigned labels, and their rect statistics
        List<Integer> assignedIndexes = new ArrayList<>();
        double maxW = -Double.MAX_VALUE, minW = Double.MAX_VALUE, meanW = 0;
        double maxH = -Double.MAX_VALUE, minH = Double.MAX_VALUE, meanH = 0;
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            if (panel.panelLabel != null && !panel.panelLabel.isEmpty())
            {
                assignedIndexes.add(i);
                int w = panel.panelRect.width, h = panel.panelRect.height;

                if (w > maxW) maxW = w;                if (w < minW) minW = w;                meanW += w;
                if (h > maxH) maxH = h;                if (h < minH) minH = h;                meanH += h;
            }
        }
        meanW = meanW / assignedIndexes.size();        meanH = meanH / assignedIndexes.size();

        //check whether labels in ascending order
        for (int i = assignedIndexes.size() - 1; i > 0; i--)
        {
            Panel panel0 = panels.get(assignedIndexes.get(i - 1));        String label0 = panel0.panelLabel;
            Panel panel1 = panels.get(assignedIndexes.get(i));            String label1 = panel1.panelLabel;
            if (label1.compareToIgnoreCase(label0) <= 0)
            {
                if (panel1.labelScore < 0.8)
                {
                    assignedIndexes.remove(i);
                }
                else
                    return false;
            }
        }

        //Reach Here: the panel labels are in ascending order, looks like RowFirst case
        //Before the first assigned panel
        for (int i = 0; i <= assignedIndexes.size(); i++)
        {
            int start = i == 0 ? 0 : assignedIndexes.get(i-1) + 1;
            int end = i == assignedIndexes.size() ? panels.size() : assignedIndexes.get(i);

            if (start >= end) continue;

            char startC, endC; //[Start, end) but [startC, endC]
            if (start == 0)
            {
                int index = assignedIndexes.get(0);
                int c = (int)panels.get(index).panelLabel.toUpperCase().charAt(0);
                startC = (char)(c - (end - start));
                if (Character.isDigit(c))
                {
                    if (startC < '0') startC = '0';
                }
                else
                {
                    if (startC < 'A') startC = 'A';
                }
            }
            else startC = (char)((int)(panels.get(start-1).panelLabel.toUpperCase().charAt(0)) + 1);

            if (end == panels.size())
            {
                int index = assignedIndexes.get(assignedIndexes.size()-1);
                int c = (int)panels.get(index).panelLabel.toUpperCase().charAt(0);
                endC = (char)(c + (end - start));
                if (Character.isDigit(c))
                {
                    if (endC > '9') endC = '9';
                }
                else
                {
                    if (endC > 'Z') endC = 'Z';
                }
            }
            else endC = (char)((int)panels.get(end).panelLabel.toUpperCase().charAt(0) - 1);

            if (end - start == (int)endC - (int)startC + 1) //Because [Start, end) but [startC, endC]
            {
                for (int j = start; j < end; j++)
                {
                    char c = (char)((int)startC + (j-start));
                    setLabelByAlignment(panels, j, c);
                }
            }
            else
            {
             //TODO: the panels and labels do not have the same number case
                return false;
            }
        }

        return true;
    }

    void setLabelByAlignment(List<Panel> panels, int j, char c)
    {
        Panel panel = panels.get(j);    //The panel to be assigned a label

        //Find the maximum aligned panel, which has labeled already
        double minRatio = Double.MAX_VALUE; Panel minPanel = null;
        for (Panel p : panels)
        {
            if (p.panelLabel == null || p.panelLabel.isEmpty()) continue;

            int leftMost = Math.min(p.panelRect.x, panel.panelRect.x);
            int rightMost = Math.max(p.panelRect.x + p.panelRect.width, panel.panelRect.x + panel.panelRect.width);
            double xRatio = (double)(rightMost - leftMost) / (double)(p.panelRect.width + panel.panelRect.width);
            int topMost = Math.min(p.panelRect.y, panel.panelRect.y);
            int bottomMost = Math.max(p.panelRect.y + p.panelRect.height, panel.panelRect.y + panel.panelRect.height);
            double yRatio = (double)(bottomMost - topMost) / (double)(p.panelRect.height + panel.panelRect.height);
            double ratio = Math.min(xRatio, yRatio);
            if (ratio < minRatio)
            {
                minPanel = p;
                minRatio = ratio;
            }
        }

        //Reach here: the maximum aligned panel is minPanel
        //We are going to add in the same location as minPanel

        //Find the closest anchor point. top-left:0, top-right:1, bottom-left:2, bottom-right:3
        Rectangle panelRect = minPanel.panelRect;
        Rectangle labelRect = minPanel.labelRect;

        Point[] panelPoints = new Point[4];
        panelPoints[0] = new Point(panelRect.x, panelRect.y);
        panelPoints[1] = new Point(panelRect.x + panelRect.width, panelRect.y);
        panelPoints[2] = new Point(panelRect.x, panelRect.y + panelRect.height);
        panelPoints[3] = new Point(panelRect.x + panelRect.width, panelRect.y + panelRect.height);

        Point[] labelPoints = new Point[4];
        labelPoints[0] = new Point(labelRect.x, labelRect.y);
        labelPoints[1] = new Point(labelRect.x + labelRect.width, labelRect.y);
        labelPoints[2] = new Point(labelRect.x, labelRect.y + labelRect.height);
        labelPoints[3] = new Point(labelRect.x + labelRect.width, labelRect.y + labelRect.height);

        int[] distances = new int[4];
        for (int i = 0; i < 4; i++)
            distances[i] = Math.abs(labelPoints[i].x - panelPoints[i].x) + Math.abs(labelPoints[i].y - panelPoints[i].y);

        int minDistance = distances[0]; int minIndex = 0;
        for (int i = 1; i < distances.length; i++)
        {
            if (distances[i] < minDistance)
            {
                minIndex = i; minDistance = distances[i];
            }
        }

        int x = 0, y = 0;
        int w = minPanel.labelRect.width;
        int h = minPanel.labelRect.height;
        switch (minIndex)
        {
            case 0:     //top-left corner
                x = panel.panelRect.x + (labelPoints[0].x - panelPoints[0].x);
                y = panel.panelRect.y + (labelPoints[0].y - panelPoints[0].y);
                break;
            case 1:     //top-right corner
                x = panel.panelRect.x + panel.panelRect.width + (labelPoints[1].x - panelPoints[1].x) - w;
                y = panel.panelRect.y + (labelPoints[0].y - panelPoints[0].y);
                break;
            case 2:     //bottom-left corner
                x = panel.panelRect.x + (labelPoints[0].x - panelPoints[0].x);
                y = panel.panelRect.y + panel.panelRect.height + (labelPoints[2].y - panelPoints[2].y) - h;
                break;
            case 3:     //bottom-right corner
                x = panel.panelRect.x + panel.panelRect.width + (labelPoints[1].x - panelPoints[1].x) - w;
                y = panel.panelRect.y + panel.panelRect.height + (labelPoints[2].y - panelPoints[2].y) - h;
                break;
        }

        panel.labelRect = new Rectangle(x, y, w, h);
        panel.panelLabel = "" + c;
    }

    Panel setLabelByProb(Panel panel, char c)
    {
        //Expand the panel rect by 100 in all 4 directions
        int l = panel.panelRect.x - 100; if ( l < 0 ) l = 0;
        int t = panel.panelRect.y - 100; if ( t < 0 ) t = 0;
        int r = panel.panelRect.x + panel.panelRect.width + 100; if (r > this.figure.imageWidth) r = this.figure.imageWidth;
        int b = panel.panelRect.y + panel.panelRect.height + 100; if (b > this.figure.imageHeight) b = this.figure.imageHeight;

        Rectangle panelRect = new Rectangle(l, t, r-l, b-t);
        //Find the label, which has the highest probabilities
        Panel maxLabel = null; double maxProb = -1.0;
        for (Panel label : figure.panelsLabelHoGSvm)
        {
            Rectangle labelRect = label.labelRect;
            if (!panelRect.intersects(labelRect)) continue;
            int index = PanelSeg.getLabelCharIndex(c);
            double prob = label.labelProbs[index];
            if (prob > maxProb)
            {
                maxLabel = label; maxProb = prob;
            }
        }

        Panel panelNew = maxLabel;
        panelNew.panelLabel = "" + c;
        panelNew.panelRect = panel.panelRect;

        return panelNew;
    }

    Panel smallestLabel(List<Panel> labels)
    {
        Panel smallest = labels.get(0);
        for (int k = 1; k < labels.size(); k++)
        {
            Panel label = labels.get(k);
            if (label.panelLabel.compareToIgnoreCase(smallest.panelLabel) < 0)
                smallest =label;
        }
        return smallest;
    }

    void handlePanelWithLabels(Panel panel, List<Panel> labelSet)
    {

    }


    void displayResults()
    {
        imshow("Image", figure.imageOriginal);

        opencv_core.Mat uniformPanels = Figure.drawAnnotation(figure.imageColor, figure.panelsUniformBand);
        imshow("Uniform Panels", uniformPanels);

        opencv_core.Mat labelPanels = Figure.drawAnnotation(figure.imageColor, figure.panelsLabelHoGSvmBeam);
        imshow("Panel Labels", labelPanels);

        opencv_core.Mat matchingResult = new opencv_core.Mat();
        opencv_core.copyMakeBorder(figure.imageColor, matchingResult, 0, 50, 0, 50, opencv_core.BORDER_CONSTANT, new opencv_core.Scalar());
        if (panels != null) {
            for (int i = 0; i < panels.size(); i++) {
                Panel panel = panels.get(i);
                List<Panel> labels = this.labelSets.get(i);
                opencv_core.Scalar color = AlgOpenCVEx.getColor(i);

                if (panel.panelRect != null && !panel.panelRect.isEmpty()) {
                    opencv_core.Rect panel_rect = AlgOpenCVEx.Rectangle2Rect(panel.panelRect);
                    opencv_imgproc.rectangle(matchingResult, panel_rect, color, 3, 8, 0);
                }
                for (int j = 0; j < labels.size(); j++) {
                    Panel label = labels.get(j);
                    opencv_core.Rect label_rect = AlgOpenCVEx.Rectangle2Rect(label.labelRect);
                    opencv_imgproc.rectangle(matchingResult, label_rect, color, 1, 8, 0);
                }
            }

            for (int i = 0; i < this.panels.size(); i++) {
                List<Panel> labels = this.labelSets.get(i);
                opencv_core.Scalar color = AlgOpenCVEx.getColor(i);
                for (int j = 0; j < labels.size(); j++) {
                    Panel label = labels.get(j);
                    opencv_core.Rect label_rect = AlgOpenCVEx.Rectangle2Rect(label.labelRect);

                    String labelStr = label.panelLabel;
                    double score = ((int) (label.labelScore * 1000 + 0.5)) / 1000.0;
                    opencv_core.Point bottom_left = new opencv_core.Point(label_rect.x() + label_rect.width(), label_rect.y() + label_rect.height() + 10);
                    opencv_imgproc.putText(matchingResult, labelStr + " " + Double.toString(score), bottom_left, opencv_imgproc.CV_FONT_HERSHEY_PLAIN, 1, color, 1, 8, false);
                }
            }
        }
        imshow("Matching Result", matchingResult);

        if (figure.structuredEdge != null)
        {
            imshow("StructuredEdges", figure.structuredEdge);
            imshow("Binary Edges", figure.binaryEdgeMap);
        }

        opencv_core.Mat segResult = new opencv_core.Mat();
        opencv_core.copyMakeBorder(figure.imageColor, segResult, 0, 50, 0, 50, opencv_core.BORDER_CONSTANT, new opencv_core.Scalar());
        if (figure.panels.size() == 1)
            segResult = Figure.drawAnnotation(segResult, figure.panels, "single");
        else if (figure.panels.size() > 1)
            segResult = Figure.drawAnnotation(segResult, figure.panels, "Multi");
        imshow("Segmentation Result", segResult);

        waitKey();
    }
}
