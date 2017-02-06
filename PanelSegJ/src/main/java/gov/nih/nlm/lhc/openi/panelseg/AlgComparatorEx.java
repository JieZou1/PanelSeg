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


