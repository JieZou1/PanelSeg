package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.commons.io.FilenameUtils;

import java.awt.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;

/**
 * Programs related to evaluation
 *
 * Created by jzou on 9/19/2016.
 */
final class ExpEval extends Exp
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if (args.length != 3) {
            System.out.println("Usage: java -cp PanelSegJ.jar ExpEval <Sample List File> <target folder> <method>");
            System.out.println("	This is a utility program to do evaluation for various panel segmentation algorithms.");
            System.out.println("method:");
            System.out.println("LabelRegHog		LabelRegHog method for Label Detection");
            System.out.println("LabelRegHogSvm		LabelRegHogSvm method for Label Recognition");
            System.exit(0);
        }

        PanelSeg.SegMethod method = null;
        switch (args[2]) {
            case "LabelRegHog":  method = PanelSeg.SegMethod.LabelRegHog; break;
            case "LabelRegHogSvm": method = PanelSeg.SegMethod.LabelRegHogSvm; break;
            default:
                System.out.println("Unknown method!!");
                System.exit(0);
        }

        ExpEval eval = new ExpEval(args[0], args[1], method);
        eval.generate();
    }

    private PanelSeg.SegMethod method;

    private HashMap<String, Integer> countIndividualLabelGT;    //The count for each individual label (ground truth)
    private HashMap<String, Integer> countIndividualLabelAuto;  //The count for each individual label (auto recognized)
    private HashMap<String, Integer> countIndividualLabelCorrect;  //The count for each individual label correctly recognized by the auto method.

    private ArrayList<ArrayList<String>> gtLabels;          //The gt panel labels for all samples
    private ArrayList<ArrayList<String>> autoLabels;
    private ArrayList<ArrayList<String>> missingLabels;
    private ArrayList<ArrayList<String>> falseAlarmLabels;

    /**
     * Ctor, set targetFolder and then collect all imagePaths
     * It also set the method
     *
     * @param trainListFile
     * @param targetFolder
     */
    private ExpEval(String trainListFile, String targetFolder, PanelSeg.SegMethod method)
    {
        super(trainListFile, targetFolder, false);
        this.method = method;
    }

    private void generate()
    {
        //Initialize
        countIndividualLabelGT = new HashMap<>();
        countIndividualLabelAuto = new HashMap<>();
        countIndividualLabelCorrect = new HashMap<>();
        autoLabels = new ArrayList<>();
        gtLabels = new ArrayList<>();
        missingLabels = new ArrayList<>();
        falseAlarmLabels = new ArrayList<>();

        for (int k = 0; k < imagePaths.size(); k++) generate(k);

        reportLabelRegEval();
    }

    void generate(int k)
    {
        ArrayList<Panel> gtPanels = null, autoPanels = null;

        Path imagePath = imagePaths.get(k);
        System.out.println(Integer.toString(k) + ": processing " + imagePath.toString());

        //Load ground truth annotation
        {
            String imageFile = imagePath.toString();
            String xmlFile = FilenameUtils.removeExtension(imageFile) + "_data.xml";

            File annotationFile = new File(xmlFile);
            try {
                gtPanels = iPhotoDraw.loadPanelSegGt(annotationFile);
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        }

        //Load segmentation results
        {
            String imageFile = imagePath.toFile().getName();
            String xmlFile = FilenameUtils.removeExtension(imageFile) + "_data.xml";
            File annotationFile  = targetFolder.resolve(xmlFile).toFile();
            try {
                autoPanels = iPhotoDraw.loadPanelLabelRegAuto(annotationFile);
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        }

        evalLabelReg(gtPanels, autoPanels);

    }

    /**
     *
     * @param gtPanels
     * @param autoPanels
     */
    private void evalLabelReg(ArrayList<Panel> gtPanels, ArrayList<Panel> autoPanels)
    {
        //Filter out gtPanels and autoPanels, which do not contain qualified panel labels.
        gtPanels = filterLabels(gtPanels);
        //autoPanels = filterLabels(autoPanels);

        {//Count Auto results and initialize counters
            ArrayList<String> autoLabel = new ArrayList<>();
            for (int i = 0; i < autoPanels.size(); i++) {
                Panel panel = autoPanels.get(i);   String label = panel.panelLabel.toLowerCase();
                autoLabel.add(label);
                if (countIndividualLabelAuto.containsKey(label))
                    countIndividualLabelAuto.put(label, countIndividualLabelAuto.get(label) + 1);
                else {
                    countIndividualLabelGT.put(label, 0);
                    countIndividualLabelAuto.put(label, 1);
                    countIndividualLabelCorrect.put(label, 0);
                }
            }
            autoLabels.add(autoLabel);
        }

        {//count GT results, and initialize counters.
            ArrayList<String> gtLabel = new ArrayList<>();
            for (int i = 0; i < gtPanels.size(); i++) {
                Panel panel = gtPanels.get(i);     String label = panel.panelLabel.toLowerCase();
                gtLabel.add(label);
                if (countIndividualLabelGT.containsKey(label))
                    countIndividualLabelGT.put(label, countIndividualLabelGT.get(label) + 1);
                else {
                    countIndividualLabelGT.put(label, 1);
                    countIndividualLabelAuto.put(label, 0);
                    countIndividualLabelCorrect.put(label, 0);
                }
            }
            gtLabels.add(gtLabel);
        }

        //Match AUTO to GT. GT label rects do not overlap for sure, but AUTO label rects may overlap
        ArrayList<Integer> indexesTaken = new ArrayList<>();
        ArrayList<String> missingLabel = new ArrayList<>();
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
                if (method == PanelSeg.SegMethod.LabelRegHog)
                { //For HoG Detection case, provided that it is detected, we count it as correct. No need to check label
                    autoLabel = gtLabel;
                }
                if (!autoLabel.equals(gtLabel)) continue;

                //REACH HERE: found a matching auto-panel
                foundIndex = one.index;
                break;
            }

            if (foundIndex >= 0)
            {   //This gt-panel is recognized
                String label = panel.panelLabel.toLowerCase();
                countIndividualLabelCorrect.put(label, countIndividualLabelCorrect.get(label) + 1);
                indexesTaken.add(foundIndex);
            }
            else
            {   //This gt-panel is missed
                missingLabel.add(panel.panelLabel);
            }
        }
        missingLabels.add(missingLabel);

        //Collect false alarms
        ArrayList<String> falseAlarmLabel = new ArrayList<>();
        for (int i = 0; i < autoPanels.size(); i++)
        {
            if (indexesTaken.indexOf(i) != -1) continue; //correct cases.
            Panel panel = autoPanels.get(i);
            falseAlarmLabel.add(panel.panelLabel);
        }
        falseAlarmLabels.add(falseAlarmLabel);
    }

    /**
     * Filter out gtPanels, which do not contain qualified panel labels.
     * @param panels
     * @return
     */
    private ArrayList<Panel> filterLabels(ArrayList<Panel> panels)
    {
        ArrayList<Panel> panelsFiltered = new ArrayList<>();
        for (int i = 0; i < panels.size(); i++) {
            Panel panel = panels.get(i);
            if (panel.labelRect == null || panel.labelRect.isEmpty())
                continue; //In this panel, there is no label.
            if (panel.panelLabel == null || panel.panelLabel.length() != 1)
                continue; //For now, we can handle single char panel label only
            panelsFiltered.add(panel);
        }
        return panelsFiltered;
    }

    private int sum(HashMap<String, Integer> count)
    {
        int sum = 0;
        for (Integer value: count.values()) sum += value;
        return sum;
    }

    private void reportLabelRegEval()
    {
        //Sum up the individual results
        int countTotalLabelsGT = sum(countIndividualLabelGT);
        int countTotalLabelsAuto = sum(countIndividualLabelAuto);
        int countTotalLabelsCorrect = sum(countIndividualLabelCorrect);

        //Save result
        try (PrintWriter pw = new PrintWriter("eval.txt"))
        {
            float precision, recall; int countGT, countAuto, countCorrect; String item;

            pw.println("Total images processed: " + imagePaths.size());
//            pw.println("Total processing time: " + (endTime - startTime)/1000.0 + " secondes.");
//            pw.println("Average processing time: " + ((endTime - startTime)/1000.0)/allPaths.size() + " secondes.");

            pw.println();
            pw.println("Item\tGT\tAuto\tCorrect\tPrecision\tRecall");

            item = "Total";
            countGT = countTotalLabelsGT; countAuto = countTotalLabelsAuto; countCorrect = countTotalLabelsCorrect;
            precision = (float)countCorrect / countAuto; precision = (float) (((int)(precision*1000+0.5))/10.0);
            recall = (float)countCorrect / countGT; recall = (float) (((int)(recall*1000+0.5))/10.0);
            pw.println(item + "\t" + countGT + "\t" + countAuto + "\t" + countCorrect + "\t" + precision + "\t" + recall);

            pw.println();

            for (String label: countIndividualLabelGT.keySet())
            {
                item = label;
                countGT = countIndividualLabelGT.get(item); countAuto = countIndividualLabelAuto.get(item); countCorrect = countIndividualLabelCorrect.get(item);
                precision = (float)countCorrect / countAuto; precision = (float) (((int)(precision*1000+0.5))/10.0);
                recall = (float)countCorrect / countGT; recall = (float) (((int)(recall*1000+0.5))/10.0);
                pw.println(item + "\t" + countGT + "\t" + countAuto + "\t" + countCorrect + "\t" + precision + "\t" + recall);
            }

            pw.println();

            //Missing Labels:
            int totalMissing = 0, totalFalseAlarm = 0;
            for (int i = 0; i < missingLabels.size(); i++) totalMissing += missingLabels.get(i).size();
            for (int i = 0; i < falseAlarmLabels.size(); i++) totalFalseAlarm += falseAlarmLabels.get(i).size();
            pw.println("Total Missing: " + totalMissing);
            pw.println("Total False Alarm: " + totalFalseAlarm);
            for (int i = 0; i < missingLabels.size(); i++)
            {
                if (missingLabels.get(i).size() == 0 && falseAlarmLabels.get(i).size() == 0) continue; //All correct, we don't care.
                pw.println(imagePaths.get(i));
                pw.print("\t" + "GT Labels:\t");	for (int k = 0; k < gtLabels.get(i).size(); k++) pw.print(gtLabels.get(i).get(k) + " "); pw.println();
                pw.print("\t" + "Missing Labels:\t");	for (int k = 0; k < missingLabels.get(i).size(); k++) pw.print(missingLabels.get(i).get(k) + " "); pw.println();
                if (method != PanelSeg.SegMethod.LabelRegHog)
                {
                    pw.print("\t" + "Auto Labels:\t");	for (int k = 0; k < autoLabels.get(i).size(); k++) pw.print(autoLabels.get(i).get(k) + " "); pw.println();
                    pw.print("\t" + "False Alarm Labels:\t"); for (int k = 0; k < falseAlarmLabels.get(i).size(); k++) pw.print(falseAlarmLabels.get(i).get(k) + " "); pw.println();
                }
            }

        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /**
     * Find panels which overlap with panelsToMatch and sort according to the percentages of overlapping in panel in descending order
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

/**
 * A class storing results of a panel compares to a list of panels. It stores:
 * index: the index of each panel in the list of panels.
 * score1: the overlapping percentage of panel panel
 * score2: the overlapping percentage of the panel in the list, whose index is index.
 */
class PanelOverlappingScore1Score2Index
{
    int index;
    Panel panel1, panel2;
    double score1, score2;

    public PanelOverlappingScore1Score2Index(int index, Panel panel1, Panel panel2, double score1, double score2)
    {
        this.index = index;
        this.score1 = score1;        this.score2 = score2;
        this.panel1 = panel1;        this.panel2 = panel2;
    }
}

/**
 * Comparator for sorting score1 of PanelOverlappingIndesScores.
 * @author Jie Zou
 */
class PanelOverlappingScore1Descending implements Comparator<PanelOverlappingScore1Score2Index>
{
    public int compare(PanelOverlappingScore1Score2Index o1, PanelOverlappingScore1Score2Index o2)
    {
        double diff = o2.score1 - o1.score1;
        if (diff > 0) return 1;
        else if (diff == 0) return 0;
        else return -1;
    }
}


