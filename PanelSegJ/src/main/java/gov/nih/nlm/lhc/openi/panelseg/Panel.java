package gov.nih.nlm.lhc.openi.panelseg;

import java.awt.Rectangle;
import java.util.Comparator;

/**
 * A simple class for holding panel segmentation result. We want to make this simple.
 * Put all complicated algorithm related stuffs into the algorithm classes.
 * The major reason for this is to separate data and algorithms.
 * Such that we could have a clean data structure for result, not embedded in the various actual algorithms.
 * This also makes serialization to XML much easier.
 *
 * Created by jzou on 8/25/2016.
 */
public class Panel
{
    Rectangle panelRect;	//The panel bounding box
    String panelLabel;		//The panel label
    Rectangle labelRect;	//The panel label bounding box

    //Not essential, but useful info about the panel.
    double labelScore;		//The confidence of the panel label
    double[] labelProbs;	//The posterior probabilities of all possible classes. Mostly used to find an optimal label set.
}
