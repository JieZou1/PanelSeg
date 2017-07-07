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
 * Comparator for sorting Panels in descending order of labelScore.
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
 * Comparator for sorting Panels in ascending order of labelScore.
 * @author Jie Zou
 */
class LabelScoreAscending implements Comparator<Panel>
{
    public int compare(Panel o1, Panel o2)
    {
        double diff = o1.labelScore - o2.labelScore;
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
class PanelRectLeftAscending implements Comparator<Panel>
{
    public int compare(Panel o1, Panel o2)
    {
        double diff = o1.panelRect.x - o2.panelRect.x;
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
class PanelRectTopAscending implements Comparator<Panel>
{
    public int compare(Panel o1, Panel o2)
    {
        double diff = o1.panelRect.y - o2.panelRect.y;
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
 * Comparator for sorting Panel Rect in Row first order
 * @author Jie Zou
 */

class PanelRectRowFirst implements Comparator<Panel>
{
    @Override
    public int compare(Panel o1, Panel o2)
    {
        int left1 = o1.panelRect.x, left2 = o2.panelRect.x;
//        int right1 = o1.panelRect.x + o1.panelRect.width, right2 = o2.panelRect.x + o2.panelRect.width;
//        int width1 = o1.panelRect.width, width2 = o2.panelRect.width;
        int top1 = o1.panelRect.y, top2 = o2.panelRect.y;
        int bottom1 = o1.panelRect.y + o1.panelRect.height, bottom2 = o2.panelRect.y + o2.panelRect.height;
        int height1 = o1.panelRect.height, height2 = o2.panelRect.height;

        if (top1 <= top2)
        {
            if (top2 - bottom1 > - height1 / 2) return -1;
            //Same row, compare x
            int diff = left1 - left2;
            if (diff > 0) return 1;
            else if (diff == 0) return 0;
            else return -1;
        }
        else
        {
            if (top1 - bottom2 > - height2 / 2) return 1;
            //Same row, compare x
            int diff = left1 - left2;
            if (diff > 0) return 1;
            else if (diff == 0) return 0;
            else return -1;
        }
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


