package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_highgui.waitKey;

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

        //No labelSets are detected, and there is only one panel.
        //This is reasonable to be a single panel figure without labelSets
        if (figure.panelsLabelHoGSvm.size() == 0 && figure.panelsUniformBand.size() == 1)
        {
            figure.panels = figure.panelsUniformBand;
        }
        else
        {   //Merge/split panels with labelSets to generate final result
            panels = new ArrayList<>(); labelSets = new ArrayList<>();
            for (int i = 0; i < figure.panelsUniformBand.size(); i++)
            {
                panels.add(figure.panelsUniformBand.get(i));
                labelSets.add(new ArrayList<>());
            }

            //Merge split-panels and recognized-panel-labelSets
            matchPanels();

            //Reach Here: panels and labelSets are merged and saved in this.panels and this.labelSets
            //1. Handle panels with matching labelSets, remove or further splitting
            //2. Handle panels without matching labelSets, merge to the closet one or adding a label
            //structuredEdgeAnalysis();
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

    void matchPanels()
    {
        //Assign labelSets to the closet panels
        for (int i = 0; i < figure.panelsLabelHoGSvm.size(); i++)
        {
            Panel label = figure.panelsLabelHoGSvm.get(i);

            //find max overlapping
            int maxIndex = -1; double maxSize = -1;
            for (int j = 0; j < panels.size(); j++)
            {
                Panel panel = panels.get(j);
                Rectangle intersection = panel.panelRect.intersection(label.labelRect);
                if (intersection.isEmpty()) continue;
                double size = intersection.width * intersection.height;
                if (size > maxSize)
                {
                    maxIndex = j; maxSize = size;
                }
            }
            if (maxIndex != -1)
            {
                labelSets.get(maxIndex).add(label);
                continue;
            }

            //No overlapping, we find the closet panel
            int minIndex = -1; int minDistance = Integer.MAX_VALUE;
            for (int j = 0; j < panels.size(); j++)
            {
                Panel panel = panels.get(j);
                int horDistance, verDistance, distance;
                if (label.labelRect.x < panel.panelRect.x)
                {
                    horDistance = panel.panelRect.x - (label.labelRect.x + label.labelRect.width);
                    if (horDistance < 0) horDistance = 0;
                }
                else
                {
                    horDistance = label.labelRect.x - (panel.panelRect.x + panel.panelRect.width);
                    if (horDistance < 0) horDistance = 0;
                }
                if (label.labelRect.y < panel.panelRect.y)
                {
                    verDistance = panel.panelRect.y - (label.labelRect.y + label.labelRect.height);
                    if (verDistance < 0) verDistance = 0;
                }
                else
                {
                    verDistance = label.labelRect.y - (panel.panelRect.y + panel.panelRect.height);
                    if (verDistance < 0) verDistance = 0;
                }
                distance = verDistance + horDistance;
                if (distance < minDistance)
                {
                    minDistance = distance; minIndex = j;
                }
            }
            labelSets.get(minIndex).add(label);
        }
    }

    void structuredEdgeAnalysis()
    {
        PanelSplitStructuredEdge splitStructuredEdge = new PanelSplitStructuredEdge(figure);
        splitStructuredEdge.detectEdges();

        //Process panels which has matching labels.
        // We either remove the extra labels, or split into more panels
        panelsT = new ArrayList<>(); labelSetsT = new ArrayList<>();
        for (int i = 0; i < this.panels.size(); i++)
        {
            Panel panel = this.panels.get(i);
            List<Panel> labelSet = this.labelSets.get(i);

            if (labelSet.size() == 0 || labelSet.size() == 1) {
                panelsT.add(panel);
                labelSetsT.add(labelSet);
                continue;
            }

            splitStructuredEdge.split(AlgOpenCVEx.Rectangle2Rect(panel.panelRect));
        }

        //complete analysis, save the result
//        panels = panelsT;
//        labelSets = labelSetsT;

        //Process panels which has no matching labels.
        //We either merge it into the existing panel or add labels

        for (int i = 0; i < this.panels.size(); i++)
        {
            Panel panel = this.panels.get(i);
            List<Panel> labelSet = this.labelSets.get(i);
        }

        //complete analysis, save the result
//        panels = panelsT;
//        labelSets = labelSetsT;
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
