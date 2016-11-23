package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FilenameUtils;

import java.awt.*;
import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;

/**
 *
 * For Label Sequence Classification
 *
 * Created by jzou on 11/22/2016.
 */
public class TrainLabelSequenceClassify extends Exp
{
    public static void main(String args[]) throws Exception {
        //Stop and print error msg if no arguments passed.
        if (args.length != 0) {
            System.out.println();

            System.out.println("Usage: java -cp PanelSegJ.jar TrainLabelSequenceClassify");
            System.out.println("Training tasks for Label Detection.");

            System.out.println();
        }

        String trainListFile, targetFolder;
        trainListFile = "\\Users\\jie\\projects\\PanelSeg\\Exp\\train.txt";
        targetFolder = "\\Users\\jie\\projects\\PanelSeg\\Exp\\PanelSeg\\Train";

        TrainLabelSequenceClassify train = new TrainLabelSequenceClassify(trainListFile, targetFolder);
        train.doWorkSingleThread();
        //train.doWorkMultiThread();
    }

    TrainLabelSequenceClassify(String trainListFile, String targetFolder)
    {
        super(trainListFile, targetFolder, false);
    }

    ArrayList<Panel> autoPanels, gtPanels; //Auto segmentation results and GT annotation
    ArrayList<Panel> correctPanels, incorrectPanels; //The correct and in correct panels from AUTO segmented Panels

    void doWork(int i)
    {
        Path imagePath = imagePaths.get(i);
        System.out.println(Integer.toString(i) + ": processing " + imagePath.toString());

        autoPanels = loadAutoAnnotation(imagePath);    //Load segmentation results
        gtPanels = loadGtAnnotation(imagePath); //Load ground truth annotation

        //Filter out gtPanels, which do not contain qualified panel labels.
        gtPanels = filterLabels(gtPanels);
        if (gtPanels.size() < 2) return; //No qualified labels, we don't use it.

        //Match AUTO to GT to separate AUTO into correct and incorrect panel labels.
        separateAutoByMatching2GT();

        //Find all possible correct and incorrect sequences
        correctPanels.sort(new PanelLabelAscending());
        incorrectPanels.sort(new PanelLabelAscending());
        collectCorrectSequences(2);
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

    private void collectCorrectSequences(int order)
    {

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
}
