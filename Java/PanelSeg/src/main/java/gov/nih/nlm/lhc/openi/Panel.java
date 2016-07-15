package gov.nih.nlm.lhc.openi;

import java.awt.Rectangle;
import java.util.Comparator;

/**
 * A simple class for holding panel segmentation result. We want to make this simple. 
 * Put all complicated algorithm related stuffs into the algorithm classes.  
 * The major reason for this is to separate data and algorithms. 
 * Such that we could have a clean data structure for result, not embedded in the various actual algorithms. 
 * This also makes serialization to XML much easier. 
 * 
 * @author Jie Zou
 *
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

/**
 * Comparator for sorting Panels in reverse order of labelScore.
 * @author Jie Zou
 */
class LabelScoreDescending implements Comparator<Panel>
{
	public int compare(Panel o1, Panel o2) 
	{
		double diff = o2.labelScore - o1.labelScore;
		if (diff > 0) return 1;
		else if (diff == 0) return 0;
		else return -1;
	}
}

/**
* Comparator for sorting Panels vertically based on the LabelRect.Left
* @author Jie Zou
*/
class LabelRectLeftAscending implements Comparator<Panel>
{
	public int compare(Panel o1, Panel o2) 
	{
		double diff = o1.labelRect.x - o2.labelRect.x;
		if (diff > 0) return 1;
		else if (diff == 0) return 0;
		else return -1;
	}
	
}

/**
 * Comparator for sorting Panels horizontally based on the LabelRect.Top
 * @author Jie Zou
 */
class LabelRectTopAscending implements Comparator<Panel>
{
	public int compare(Panel o1, Panel o2) 
	{
		double diff = o1.labelRect.y - o2.labelRect.y;
		if (diff > 0) return 1;
		else if (diff == 0) return 0;
		else return -1;
	}
	
}

/**
 * Comparator for sorting Panels based on the panelLabel. The case is ignored.
 * @author Jie Zou
 */
class PanelLabelAscending implements Comparator<Panel>
{
	public int compare(Panel o1, Panel o2) 
	{
		int diff = o1.panelLabel.toLowerCase().compareTo(o2.panelLabel.toLowerCase());
		if (diff > 0) return 1;
		else if (diff == 0) return 0;
		else return -1;
	}
}

