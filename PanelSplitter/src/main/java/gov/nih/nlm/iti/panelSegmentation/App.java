package gov.nih.nlm.iti.panelSegmentation;

import weka.classifiers.Classifier;

import java.util.ArrayList;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main(String args[]) throws Exception{
        PanelSplitter ps = new PanelSplitter();

        String ocrModelFileName = "/Users/jie/projects/PanelSeg/programs/PanelSplitter/models/NN_OCR.model";
        Classifier OCR_model = null;
        try {
            OCR_model = (Classifier) weka.core.SerializationHelper.read(ocrModelFileName);
        } catch (Exception e) {
            System.out.println("Unable to read the OCR model file");
            e.printStackTrace();
        }

        System.out.println("OCR model Loaded");

//		String imgName = "\\\\lhcdevfiler\\cebchest-imaging\\Experiments_Personal\\santosh\\2Xue\\test1\\1297-9686-41-46-2.jpg";
        String imgName = "/Users/jie/projects/PanelSeg/programs-others/Daekeun/PanelSplitter/jar/multipanel_images/1477-7819-6-23-1.jpg";

        ArrayList<PanelSplitter.Final_Panel> finalSplitPanels = ps.panelSplitter(imgName, OCR_model);

        System.out.println("Panel Splitter completes");

        for (int i = 0; i < finalSplitPanels.size(); i++){
            PanelSplitter.Final_Panel panel = finalSplitPanels.get(i);
            System.out.println("left: "+ Integer.toString(panel.left) + " right: "+ Integer.toString(panel.right)
                    + " top: "+ Integer.toString(panel.top) + " bottom: "+ Integer.toString(panel.bottom));
        }

        System.out.println("Completes: Number of panels is " + finalSplitPanels.size());
    }
}
