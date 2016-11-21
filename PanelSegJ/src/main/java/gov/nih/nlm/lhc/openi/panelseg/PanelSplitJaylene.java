package gov.nih.nlm.lhc.openi.panelseg;

import gov.nih.nlm.iti.panelSegmentation.regular.segmentation.PanelSplitter;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;

/**
 * ReFactored from Jaylene's implementation
 *
 * Created by jzou on 11/18/2016.
 */
public class PanelSplitJaylene
{
    Figure figure;

    PanelSplitJaylene(Figure figure) {
        this.figure = figure;
    }

    void split()
    {
        BufferedImage bufferedImage = AlgOpenCVEx.mat2BufferdImg(figure.imageOriginal);
        PanelSplitter extractPanel = new PanelSplitter(bufferedImage);	//Construct Jaylene's panel object for calling her segmentation method

        extractPanel.removeLabel();
        ArrayList<Rectangle> rects = extractPanel.extract();

        for (Rectangle rect : rects)
        {
            Panel panel = new Panel();
            panel.panelRect = rect;
            figure.panels.add(panel);
        }
    }
}
