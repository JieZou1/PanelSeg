package gov.nih.nlm.lhc.openi.panelseg;

import java.awt.*;
import java.util.ArrayList;

/**
 * The base class for all panel label recognition algorithms. <p>
 * This class is not intended to be instantiated, so we make it abstract.
 *
 * Created by jzou on 8/25/2016.
 */
public abstract class PanelSegLabelReg extends PanelSeg
{
    /**
     * Sort candidates and remove largely overlapped candidates. <p>
     * Largely here means the overlapping area is over half of the area of the candidates which have higher scores.
     * @param candidates
     * @return
     */
    protected ArrayList<Panel> RemoveOverlappedCandidates(ArrayList<Panel> candidates)
    {
        if (candidates == null || candidates.size() == 0 || candidates.size() == 1) return candidates;

        candidates.sort(new LabelScoreDescending());

        //Remove largely overlapped candidates
        ArrayList<Panel> results = new ArrayList<>();        results.add(candidates.get(0));
        for (int j = 1; j < candidates.size(); j++)
        {
            Panel obj = candidates.get(j);            Rectangle obj_rect = obj.labelRect;
            double obj_area = obj_rect.width * obj_rect.height;

            //Check with existing ones, if significantly overlapping with existing ones, ignore
            Boolean overlapping = false;
            for (int k = 0; k < results.size(); k++)
            {
                Rectangle result_rect = results.get(k).labelRect;
                Rectangle intersection = obj_rect.intersection(result_rect);
                if (intersection.isEmpty()) continue;
                double intersection_area = intersection.width * intersection.height;
                double result_area = result_rect.width * result_rect.height;
                if (intersection_area > obj_area / 2 || intersection_area > result_area / 2)
                {
                    overlapping = true; break;
                }
            }
            if (!overlapping) results.add(obj);
        }
        return results;
    }

}
