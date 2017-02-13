package gov.nih.nlm.lhc.openi.panelseg;

import java.awt.*;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Programs related to evaluation
 *
 * Created by jzou on 2/13/2017.
 */
public class ExpEval extends Exp
{
    public static void main(String args[])
    {
        ExpEval expPanelSeg = new ExpEval();
        try
        {
            expPanelSeg.loadProperties();
            expPanelSeg.initialize();
            expPanelSeg.doWork();
            log.info("Completed!");
        }
        catch (Exception ex)
        {
            log.error(ex.getMessage());
        }
    }

    private PanelSeg.Method method;
    private String propEvalFile;

    private HashMap<String, Integer> countIndividualLabelGT;    //The count for each individual label (ground truth)
    private HashMap<String, Integer> countIndividualLabelAuto;  //The count for each individual label (auto recognized)
    private HashMap<String, Integer> countIndividualLabelCorrect;  //The count for each individual label correctly recognized by the auto method.

    private ArrayList<ArrayList<String>> gtLabels;          //The gt panel labels for all samples
    private ArrayList<ArrayList<String>> autoLabels;
    private ArrayList<ArrayList<String>> missingLabels;
    private ArrayList<ArrayList<String>> falseAlarmLabels;

    @Override
    void loadProperties() throws Exception
    {
        loadProperties("ExpEval.properties");

        String propMethod = getProperty("Method");
        switch (propMethod)
        {
            case "LabelDetHog":
                method = PanelSeg.Method.LabelDetHog;
                break;
            case "LabelRegHogSvm":
                method = PanelSeg.Method.LabelRegHogSvm;
                break;
            case "LabelRegHogSvmThreshold":
                method = PanelSeg.Method.LabelRegHogSvmThreshold;
                break;
            case "LabelRegHogSvmBeam":
                method = PanelSeg.Method.LabelRegHogSvmBeam;
                break;

            case "LabelDetHogLeNet5":
                method = PanelSeg.Method.LabelDetHogLeNet5;
                break;
            case "LabelRegHogLeNet5Svm":
                method = PanelSeg.Method.LabelRegHogLeNet5Svm;
                break;
            case "LabelRegHogLeNet5SvmBeam":
                method = PanelSeg.Method.LabelRegHogLeNet5SvmBeam;
                break;
            case "LabelRegHogLeNet5SvmAlignment":
                method = PanelSeg.Method.LabelRegHogLeNet5SvmAlignment;
                break;

            case "PanelSplitSantosh":
                method = PanelSeg.Method.PanelSplitSantosh;
                break;
            case "PanelSplitJaylene":
                method = PanelSeg.Method.PanelSplitJaylene;
                break;
            default: throw new Exception(propMethod + " is Unknown");
        }

        propListFile = getProperty("ListFile");
        propTargetFolder = getProperty("TargetFolder");
        propEvalFile = getProperty("EvalFile");

        waitKeyContinueOrQuit("Configuration Okay? Press any key to continue, press N to quit");
    }

    @Override
    void initialize() throws Exception {
        setListFile();
        setTargetFolder(false);

        countIndividualLabelGT = new HashMap<>();
        countIndividualLabelAuto = new HashMap<>();
        countIndividualLabelCorrect = new HashMap<>();
        autoLabels = new ArrayList<>();
        gtLabels = new ArrayList<>();
        missingLabels = new ArrayList<>();
        falseAlarmLabels = new ArrayList<>();
    }

    @Override
    void doWork() throws Exception
    {
        doWorkSingleThread();
        reportLabelRegEval();
    }

    @Override
    void doWork(int k) throws Exception {
        super.doWork(k);

        Path imagePath = imagePaths.get(k);

        ArrayList<Panel> gtPanels = loadGtAnnotation(imagePath);        //Load ground truth annotation
        ArrayList<Panel> autoPanels = loadAutoAnnotation(imagePath);    //Load segmentation results

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
                if (    method == PanelSeg.Method.LabelDetHog
                        || method == PanelSeg.Method.LabelDetHogLeNet5
                    //|| method == PanelSeg.Method.LabelRegHogSvm
                        )
                { //For LabelDetHog and LabelDetHogLeNet5 cases, provided that label is detected, we count it as correct. No need to check label
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
        try (PrintWriter pw = new PrintWriter(propEvalFile))
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
            pw.println();

            for (int i = 0; i < missingLabels.size(); i++)
            {
                if (missingLabels.get(i).size() == 0 && falseAlarmLabels.get(i).size() == 0) continue; //All correct, we don't care.
                pw.println(imagePaths.get(i));
                pw.print("\t" + "GT Labels:\t");	for (int k = 0; k < gtLabels.get(i).size(); k++) pw.print(gtLabels.get(i).get(k) + " "); pw.println();
                if (missingLabels.get(i).size() > 0)
                {
                    pw.print("\t" + "Missing Labels:\t");	for (int k = 0; k < missingLabels.get(i).size(); k++) pw.print(missingLabels.get(i).get(k) + " "); pw.println();
                }
                if (method != PanelSeg.Method.LabelDetHog)
                {
                    pw.print("\t" + "Auto Labels:\t");	for (int k = 0; k < autoLabels.get(i).size(); k++) pw.print(autoLabels.get(i).get(k) + " "); pw.println();
                    if (falseAlarmLabels.get(i).size() > 0)
                    {
                        pw.print("\t" + "False Alarm Labels:\t"); for (int k = 0; k < falseAlarmLabels.get(i).size(); k++) pw.print(falseAlarmLabels.get(i).get(k) + " "); pw.println();
                    }
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
