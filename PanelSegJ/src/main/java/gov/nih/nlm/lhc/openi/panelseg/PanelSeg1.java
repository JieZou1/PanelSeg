package gov.nih.nlm.lhc.openi.panelseg;

import javafx.scene.layout.Pane;
import org.bytedeco.javacpp.opencv_core;

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
    List<Panel> panelsUniformBand;  //Panel candidates based on uniform bands
    List<Panel> panelsLabelHoGSvm;  //Panel label candidates based on HoG detection + LeNet5 + SVM

    List<Panel> panels;         //The panels
    List<List<Panel>> labels;   //The list of labels assigned to each of the panels

    static void initialize(Properties properties) throws Exception
    {
        LabelDetectHog.initialize(properties);
        LabelClassifyLeNet5.initialize(properties);
        LabelClassifyHogSvm.initialize(properties);
    }

    PanelSeg1(Figure figure) {this.figure = figure; }

    void segment() throws Exception
    {
        //Split based on Uniform Bands
        panelsUniformBand = splitUniformBands();

        //Label Recognize using HoG + SVM
        panelsLabelHoGSvm = labelRegHogSvm();

        //Merge split panels and recognized panel labels
        mergePanels();

        displayResults();
    }

    List<Panel> splitUniformBands()
    {
        Figure fig = new Figure(figure.imageOriginal);
        PanelSplitJaylene splitJaylene = new PanelSplitJaylene(fig);
        splitJaylene.split();
        return splitJaylene.figure.getSegResultWithPadding();
    }

    List<Panel> labelRegHogSvm() throws Exception
    {
        Figure fig = new Figure(figure.imageOriginal);

        LabelDetectHog detectHog = new LabelDetectHog(fig);
        detectHog.hoGDetect();        //HoG Detection, detected patches are stored in hogDetectionResult
        detectHog.mergeDetectedLabelsSimple();  //Merge all hogDetectionResult to panels

        LabelClassifyLeNet5 classifyLeNet5 = new LabelClassifyLeNet5(fig);
        classifyLeNet5.LeNet5Classification();    //LeNet5 classification of each detected patch in figure.panels.
        classifyLeNet5.removeFalseAlarms(0.02);

        //Do label classification with HoG-SVM
        LabelClassifyHogSvm classifySvm = new LabelClassifyHogSvm(fig);
        classifySvm.svmClassificationWithLeNet5();
        classifySvm.mergeRecognitionLabelsSimple();

        return fig.getSegResultWithPadding();
    }

    void mergePanels()
    {
        //Assign labels to the closet panels
        panels = new ArrayList<>(); labels = new ArrayList<>();
        for (int i = 0; i < panelsUniformBand.size(); i++)
        {
            panels.add(panelsUniformBand.get(i));
            labels.add(new ArrayList<>());
        }

        for (int i = 0; i < panelsLabelHoGSvm.size(); i++)
        {
            Panel label = panelsLabelHoGSvm.get(i);

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
                labels.get(maxIndex).add(label);
                continue;
            }

            //No overlapping, we find the closet panel

        }
    }

    void displayResults()
    {
        imshow("Image", figure.imageOriginal);

        opencv_core.Mat uniformPanels = Figure.drawAnnotation(figure.imageColor, panelsUniformBand);
        imshow("Uniform Panels", uniformPanels);

        opencv_core.Mat labelPanels = Figure.drawAnnotation(figure.imageColor, panelsLabelHoGSvm);
        imshow("Panel Labels", labelPanels);

        waitKey();
    }
}
