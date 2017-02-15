package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;

import java.awt.*;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * Label Sequence Classification Training codes
 *
 * Created by jzou on 2/15/2017.
 */
public class ExpTrainLabelSeqClassifySvm extends Exp
{
    public static void main(String args[])
    {
        log.info("Training tasks for SVM Label Sequence Classification.");

        ExpTrainLabelSeqClassifySvm exp = new ExpTrainLabelSeqClassifySvm();
        try
        {
            exp.loadProperties();
            exp.initialize();
            exp.doWork();
            log.info("Completed!");
        }
        catch (Exception ex)
        {
            log.error(ex.getMessage());
        }
    }

    private opencv_core.Mat image;
    private ArrayList<Panel> autoPanels, gtPanels; //Auto segmentation results and GT annotation
    private ArrayList<Panel> correctPanels, incorrectPanels; //The correct and in correct panels from AUTO segmented Panels

    private ArrayList<float[]> featuresAll; //All features of an order
    private ArrayList<Double> labelsAll; //All labels of an order
    private ArrayList<ArrayList<float[]>> featuresAllOrder; //All features of all order (2 or 3, ..., or 26)
    private ArrayList<ArrayList<Double>> labelsAllOrder; //All labels of all order (2 or 3, ..., or 26)

    @Override
    void loadProperties() throws Exception
    {
        loadProperties("ExpTrainLabelSeqClassifySvm.properties");

        propTrainFolder = getProperty("TrainFolder");
        propTargetFile = getProperty("TargetFile");

        waitKeyContinueOrQuit("Configuration Okay? Press any key to continue, press N to quit");
    }

    @Override
    void initialize() throws Exception
    {
        setListFile();
        setTargetFolder(false);

        featuresAllOrder = new ArrayList<>();
        labelsAllOrder = new ArrayList<>();
        for (int i = 0; i < 26; i++)
        {
            featuresAllOrder.add(new ArrayList<>());
            labelsAllOrder.add(new ArrayList<>());
        }
    }

    @Override
    void doWork() throws Exception {
        doWorkSingleThread();
        save();
    }

    void doWork(int i)
    {
        Path imagePath = imagePaths.get(i);
        System.out.println(Integer.toString(i) + ": processing " + imagePath.toString());

        //if (!imagePath.toString().endsWith("PMC4402627_ejhg2014150f1.jpg")) return;

        image = imread(imagePath.toString());

        autoPanels = loadAutoAnnotation(imagePath);    //Load segmentation results
        gtPanels = loadGtAnnotation(imagePath); //Load ground truth annotation

        //Filter out gtPanels, which do not contain qualified panel labels.
        gtPanels = filterLabels(gtPanels);
        if (gtPanels.size() < 2) return; //No qualified labels, we don't use it.

        //Match AUTO to GT to separate AUTO into correct and incorrect panel labels.
        separateAutoByMatching2GT();

        //Feature Extraction from correct and incorrect sequences
        for (int order = 2; order <= 6; order++)
        {
            featureExtraction(order);
            if (featuresAll.size() > 0)
            {
                featuresAllOrder.get(order).addAll(featuresAll);
                labelsAllOrder.get(order).addAll(labelsAll);
            }
        }
    }

    private void separateAutoByMatching2GT()
    {
        //Match AUTO to GT. GT label rects do not overlap for sure, but AUTO label rects may overlap
        ArrayList<Integer> indexesTaken = new ArrayList<>();
        for (int i = 0; i < gtPanels.size(); i++)
        {
            Panel panel = gtPanels.get(i);

            //Sort all autoPanels according to its overlapping to this gt-panel.
            ArrayList<PanelOverlappingScore1Score2Index> sorted = sortOverlappingRects(panel, autoPanels);

            int foundIndex = -1;
            for (int j = 0; j < sorted.size(); j++)
            {
                PanelOverlappingScore1Score2Index one = sorted.get(j);
                Panel gtPanel = one.panel1, autoPanel = one.panel2;

                //Check overlapping score1, it has to be >= 0.25.
                //It means the overlapping area must be larger than 1/4 of the gt-panel.
                if (one.score1 < 0.25) break; //Since score1 has been sorted, if it is less, we are done.

                //Check overlapping score2, it has to be >= 0.1
                //It means the overlapping area must be larger than 1/10 of the auto-panel.
                //If it is not, the auto-panel is too large.
                if (one.score2 < 0.1) continue;

                //check whether the auto-panel has been matched to another gt-panel already
                if (indexesTaken.indexOf(one.index) != -1) continue; //This index has been matched to a gtPanel already.

                //check label
                String autoLabel = autoPanel.panelLabel.toLowerCase();
                String gtLabel = gtPanel.panelLabel.toLowerCase();
                if (!autoLabel.equals(gtLabel)) continue;

                //REACH HERE: found a matching auto-panel
                foundIndex = one.index;
                break;
            }
            if (foundIndex >= 0)
            {   //This gt-panel is recognized
                indexesTaken.add(foundIndex);
            }
        }

        correctPanels = new ArrayList<>();        incorrectPanels = new ArrayList<>();
        for (int i = 0; i < autoPanels.size(); i++)
        {
            Panel panel = autoPanels.get(i);
            if (indexesTaken.indexOf(i) == -1)
                incorrectPanels.add(panel);
            else correctPanels.add(panel);
        }
    }

    private void featureExtraction(int order)
    {
        featuresAll = new ArrayList<>();       labelsAll = new ArrayList<>(); //Reset features and labels

        List<float[]> posFeatures = new ArrayList<>();
        {   //Feature Extract for correct sequence
            if (correctPanels.size() < order) return;

            List<List<Panel>> selectedPanels = AlgMiscEx.randomItemSets(correctPanels, order,100);

            for (int i = 0; i < selectedPanels.size(); i++)
            {
                List<Panel> panelList = selectedPanels.get(i);
                panelList.sort(new PanelLabelAscending());
                Panel[] panels = panelList.toArray(new Panel[panelList.size()]);
                float[] feature = LabelSequenceClassify.featureExtraction(image, panels);
                posFeatures.add(feature);
            }
        }

        List<float[]> negFeatures = new ArrayList<>();
        if (    incorrectPanels.size() != 0 &&  //If no incorrect panels, we are not able to compile incorrect sets.
                correctPanels.size() + incorrectPanels.size() >= order &&
                posFeatures.size() != 0) //If we do not find any correct pairs, we do not compile any incorrect sets
        {   //Feature Extract for incorrect sequence
            for (int i = 1; i <= order; i++)
            {
                int incorrectCount = i, correctCount = order - i;
                if (correctPanels.size() < correctCount) continue;
                if (incorrectPanels.size() < incorrectCount) continue;

                List<List<Panel>> selectedCorrectPanels = AlgMiscEx.randomItemSets(correctPanels, correctCount, posFeatures.size());
                List<List<Panel>> selectedIncorrectPanels = AlgMiscEx.randomItemSets(incorrectPanels, incorrectCount, posFeatures.size());

                for (int j = 0; j < selectedIncorrectPanels.size(); j++)
                {
                    List<Panel> incorrects = selectedIncorrectPanels.get(j);

                    if (selectedCorrectPanels == null)
                    {
                        incorrects.sort(new PanelLabelAscending());

                        Panel[] panelArr = incorrects.toArray(new Panel[incorrects.size()]);
                        if (!LabelSequenceClassify.noDuplicateLabels(panelArr)) continue;
                        if (!LabelSequenceClassify.noOverlappingRect(panelArr)) continue;

                        float[] feature = LabelSequenceClassify.featureExtraction(image, panelArr);
                        negFeatures.add(feature);
                    }
                    else
                    {
                        for (int k = 0; k < selectedCorrectPanels.size(); k++)
                        {
                            List<Panel> corrects = selectedCorrectPanels.get(k);

                            List<Panel> panels = new ArrayList<>();
                            panels.addAll(corrects); panels.addAll(incorrects);
                            panels.sort(new PanelLabelAscending());

                            Panel[] panelArr = panels.toArray(new Panel[panels.size()]);
                            if (!LabelSequenceClassify.noDuplicateLabels(panelArr)) continue;
                            if (!LabelSequenceClassify.noOverlappingRect(panelArr)) continue;

                            float[] feature = LabelSequenceClassify.featureExtraction(image, panelArr);
                            negFeatures.add(feature);
                        }
                    }
                }
            }
        }

        //Merge Positives and Negatives
        featuresAll.addAll(posFeatures);
        for (int i = 0; i < posFeatures.size(); i++)    labelsAll.add(1.0);

        if (negFeatures.size() > posFeatures.size())
        {
            Collections.shuffle(negFeatures, new Random());
            negFeatures = negFeatures.subList(0, posFeatures.size());
        }
        featuresAll.addAll(negFeatures);
        for (int i = 0; i < negFeatures.size(); i++)    labelsAll.add(0.0);
    }

    /**
     * Find panels which overlap with panelsToMatch and sort
     * according to the percentages of overlapping in panel in descending order
     * @param panel
     * @param panelsToMatch
     */
    private ArrayList<PanelOverlappingScore1Score2Index> sortOverlappingRects(Panel panel, ArrayList<Panel> panelsToMatch)
    {
        ArrayList<PanelOverlappingScore1Score2Index> sorted = new ArrayList<>();

        for (int i = 0; i < panelsToMatch.size(); i++)
        {
            Panel panelToMatch = panelsToMatch.get(i);

            Rectangle intersect = panel.labelRect.intersection(panelToMatch.labelRect);
            double area_intersect = intersect.isEmpty() ? 0 : intersect.width * intersect.height;
            double area_panel = panel.labelRect.width * panel.labelRect.height;
            double area_panel_to_match = panelToMatch.labelRect.width * panelToMatch.labelRect.height;

            double score1 = area_intersect / area_panel;
            double score2 = area_intersect / area_panel_to_match;

            PanelOverlappingScore1Score2Index one = new PanelOverlappingScore1Score2Index(i, panel, panelToMatch, score1, score2);
            sorted.add(one);
        }

        sorted.sort(new PanelOverlappingScore1Descending());
        return sorted;
    }

    private void save() throws Exception
    {
        for (int k = 0; k < 26; k++)
        {
            ArrayList<float[]> features = featuresAllOrder.get(k);
            if (features == null || features.size() == 0) continue;
            ArrayList<Double> labels = labelsAllOrder.get(k);

            //Normalize data
            int n_feature = features.get(0).length; int n_sample = features.size();
            float[] min = new float[n_feature], max = new float[n_feature];
            for (int i = 0; i < n_feature; i++) { min[i] = Float.POSITIVE_INFINITY; max[i] = Float.NEGATIVE_INFINITY; }
            for (int i = 0; i < n_sample; i++)
            {
                for (int j = 0; j < n_feature; j++)
                {
                    if (features.get(i)[j] > max[j]) max[j] = features.get(i)[j];
                    if (features.get(i)[j] < min[j]) min[j] = features.get(i)[j];
                }
            }
            for (int i = 0; i < n_sample; i++)
                for (int j = 0; j < n_feature; j++)
                    features.get(i)[j] = (features.get(i)[j] - min[j]) / (max[j] - min[j]);

            LibSvmEx.SaveInLibSVMFormat("train" + k + ".txt", labels, features);

            try (PrintWriter pw = new PrintWriter("scaling" + k + ".txt"))
            {
                for (int i = 0; i < max.length; i++)
                    pw.println(min[i] + " " + max[i]);
            }
        }
    }
}
