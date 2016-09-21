package gov.nih.nlm.lhc.openi.panelseg;

import java.util.Comparator;
import java.awt.*;

/**
 * Comparator to sort Rectangle according to its Top in Ascending order
 * @author Jie Zou
 */
class RectangleTopAscending implements Comparator<Rectangle>
{
    public int compare(Rectangle o1, Rectangle o2)
    {
        double diff = o1.y - o2.y;
        if (diff > 0) return 1;
        else if (diff == 0) return 0;
        else return -1;
    }
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


