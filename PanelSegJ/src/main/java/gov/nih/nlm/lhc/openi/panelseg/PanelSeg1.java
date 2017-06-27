package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;

import java.awt.*;
import java.util.*;
import java.util.List;

import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_highgui.waitKey;
import static org.bytedeco.javacpp.opencv_imgproc.CV_FILLED;
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
        figure.panelsLabelHoGSvm = labelRegHogSvm();

        if (figure.panelsLabelHoGSvm.size() == 0 && figure.panelsUniformBand.size() == 1)
        {
            //No labelSets are detected, and there is only one panel.
            //This is reasonable to be a single panel figure without labelSets
            figure.panels = figure.panelsUniformBand;
        }
        else
        {   //Merge/split panels with labelSets to generate final result
            this.panels = new ArrayList<>();
            this.panels.addAll(figure.panelsUniformBand);

            //Merge split-panels and recognized-panel-labelSets
            this.labelSets = matchPanels(panels, figure.panelsLabelHoGSvm); //each panel in panels corresponds to an element in labelSets

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

        displayResults();
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
        tryRowFirst(panelsT);

        //complete analysis, save the result
        figure.panels = panelsT;
    }

    void tryRowFirst(List<Panel> panels)
    {
        panels.sort(new PanelRectRowFirst());
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

        opencv_core.Mat labelPanels = Figure.drawAnnotation(figure.imageColor, figure.panelsLabelHoGSvm);
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
