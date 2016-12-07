package gov.nih.nlm.lhc.openi.panelseg;

/**
 * Created by jzou on 11/22/2016.
 */

import com.sun.org.apache.regexp.internal.RE;
import org.bytedeco.javacpp.opencv_core;

import java.awt.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Label sequence finding using Beam Search
 * Combining both individual label probs and label sequence joint probs.
 *
 * Created by jzou on 11/22/2016.
 */
final class LabelBeamSearchAlignment
{
    private Figure figure;

    LabelBeamSearchAlignment(Figure figure) {
        this.figure = figure;
    }

    void search()
    {
        List<Panel> panels = figure.panels;
        if (panels.size() == 0) return;

        Panel.sortPanelsByNonNegProbs(panels); //Sort panels according to their non-neg probs

        //We search alignment on panels which have high prob to be labels only.
        List<Panel> candidates = new ArrayList<>();
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            if (panel.labelScore < 0.9) break;
            candidates.add(panel);
        }

        if (candidates.size() == 0) {figure.panels = candidates; return; }

        //Split the candidates into alignment sets
        List<List<Panel>> sets = new ArrayList<>();
        while (candidates.size() != 0)
        {
            List<Panel> set = new ArrayList<>();
            List<Panel> left = new ArrayList<>();
            set.add(candidates.get(0));
            for (int i = 1; i < candidates.size(); i++)
            {
                Panel candidate = candidates.get(i);
                if (Panel.aligned(candidate, set)) set.add(candidate);
                else left.add(candidate);
            }
            sets.add(set);
            candidates = left;
        }

        //Sort all sets according their sizes (larger first)
        Collections.sort(sets, new Comparator<List>(){
            public int compare(List a1, List a2) {
                return a2.size() - a1.size(); // we want biggest to smallest
            }
        });

        //For now, we pick the longest set only
        candidates = sets.get(0);
        for (int i = 0; i < candidates.size(); i++) Panel.setLabelByNonNegProbs(candidates.get(i));



        figure.panels = candidates;

    }


}