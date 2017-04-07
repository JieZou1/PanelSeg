package gov.nih.nlm.lhc.openi.panelseg;

import libsvm.svm;
import libsvm.svm_model;
import org.apache.commons.lang.ArrayUtils;
import org.bytedeco.javacpp.opencv_core;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.io.*;
import java.util.*;
import java.util.List;

/**
 * To classify label sequences
 *
 * Created by jzou on 11/23/2016.
 */
public class LabelSequenceClassify
{
    protected static final Logger log = LoggerFactory.getLogger(LabelSequenceClassify.class);

    private static String propLabelSeqSvmModels;
    static svm_model[] svmModels = null;
    static float[][] mins = null;
    static float[][] ranges = null;

    static void initialize(String propLabelSeqSvmModels) throws Exception
    {
        svmModels = new svm_model[26];
        mins = new float[26][]; ranges = new float[26][];

        String[] models = propLabelSeqSvmModels.split(";");
        for (int i = 2; i < models.length + 2; i++)
        {
            String model = models[i-2];
            String[] modelParts = model.split(",");
            String svm = modelParts[0], scale = modelParts[1];

            {   //load svm model
                InputStream modelStream = LabelSequenceClassify.class.getClassLoader().getResourceAsStream(svm);
                BufferedReader br = new BufferedReader(new InputStreamReader(modelStream));
                svmModels[i] = libsvm.svm.svm_load_model(br);
                log.info(svm + " is loaded. nr_class is " + svmModels[i].nr_class);
            }

            {   //load scale
                InputStream scaleStream = LabelSequenceClassify.class.getClassLoader().getResourceAsStream(scale);
                String line;
                List<String> lines = new ArrayList<>();
                try (BufferedReader br = new BufferedReader(new InputStreamReader(scaleStream)))
                {
                    while ((line = br.readLine()) != null) lines.add(line);
                }
                mins[i] = new float[lines.size()];
                ranges[i] = new float[lines.size()];
                for (int k = 0; k < lines.size(); k++) {
                    String[] words = lines.get(k).split("\\s+");
                    float min = Float.parseFloat(words[0]), max = Float.parseFloat(words[1]);
                    mins[i][k] = min;
                    ranges[i][k] = max - min;
                }
            }
        }
    }

    static void initialize(Properties properties) throws Exception
    {
        propLabelSeqSvmModels = properties.getProperty("LabelSeqSvmModels");
        if (propLabelSeqSvmModels == null) throw new Exception("ERROR: LabelSeqSvmModels property is Missing.");

        initialize(propLabelSeqSvmModels);
    }

    static boolean noDuplicateLabels(LabelBeamSearch.BeamItem item)
    {
        ArrayList<Character> labels = new ArrayList<>();
        for (int i = 0; i < item.labelIndexes.size(); i++)
        {
            int labelIndex = item.labelIndexes.get(i);
            if (labelIndex == PanelSeg.labelChars.length) continue; //None label patch, we allow duplicates of course.

            char ch = PanelSeg.labelChars[labelIndex];
            labels.add(ch);
        }
        return noDuplicateLabels(labels);
    }

    static boolean noDuplicateLabels(ArrayList<Character> labels)
    {
        List<Character> charsLower = new ArrayList<>();
        for (int i = 0; i < labels.size(); i++)
        {
            char chLower = Character.toLowerCase(labels.get(i));
            if (charsLower.indexOf(chLower) != -1) return false;
            charsLower.add(chLower);
        }
        return true;
    }

    static boolean noDuplicateLabels(Panel[] panels)
    {
        ArrayList<Character> labels = new ArrayList<>();
        for (int i = 0; i < panels.length; i++)
        {
            char ch = panels[i].panelLabel.charAt(0);
            labels.add(ch);
        }
        return noDuplicateLabels(labels);
    }

    static boolean noDuplicateLabels(Panel panel, List<Panel> panels)
    {
        ArrayList<Character> labels = new ArrayList<>();
        for (int i = 0; i < panels.size(); i++)
        {
            labels.add(panels.get(i).panelLabel.charAt(0));
        }
        labels.add(panel.panelLabel.charAt(0));
        return noDuplicateLabels(labels);
    }

    static boolean noOverlappingRect(LabelBeamSearch.BeamItem item, Figure figure)
    {
        //Collect all label rects.
        List<Rectangle> rectangles = new ArrayList<>();
        for (int i = 0; i < item.labelIndexes.size(); i++)
        {
            int index = item.labelIndexes.get(i);
            if (index == PanelSeg.labelChars.length) continue; //classified as negative don't care.

            Panel panel = figure.panels.get(i);
            rectangles.add(panel.labelRect);
        }

        return noOverlappingRect(rectangles);
    }

    static boolean noOverlappingRect(Panel[] panels)
    {
        //Collect all label rects.
        List<Rectangle> rectangles = new ArrayList<>();
        for (int i = 0; i < panels.length; i++)
        {
            Panel panel = panels[i];
            rectangles.add(panel.labelRect);
        }

        return noOverlappingRect(rectangles);
    }

    static boolean noOverlappingRect(List<Rectangle> rectangles)
    {
        for (int i = 0; i < rectangles.size(); i++)
        {
            Rectangle r1 = rectangles.get(i);
            for (int j = i+ 1; j < rectangles.size(); j++)
            {
                Rectangle r2 = rectangles.get(j);
                Rectangle intersect = r1.intersection(r2);
                if (!intersect.isEmpty()) return false;
            }
        }

        return true;
    }

    static boolean noOverlappingRect(Panel panel, List<Panel> panels)
    {
        List<Rectangle> rectangles = new ArrayList<>();
        for (int i = 0; i < panels.size(); i++)
        {
            rectangles.add(panels.get(i).labelRect);
        }
        rectangles.add(panel.labelRect);
        return noOverlappingRect(rectangles);
    }

    enum SequenceType {Unknown, Upper, Lower, Digit}

    static boolean sameCaseLabels(LabelBeamSearch.BeamItem item)
    {
        SequenceType typeSeq = SequenceType.Unknown;
        for (int i = 0; i < item.labelIndexes.size(); i++)
        {
            int labelIndex = item.labelIndexes.get(i);
            if (labelIndex == PanelSeg.labelChars.length) continue; //None label patch

            char ch = PanelSeg.labelChars[labelIndex];

            SequenceType type;
            if (Character.isDigit(ch))          type = SequenceType.Digit;
            else if (Character.isLowerCase(ch)) type = SequenceType.Lower;
            else                                type = SequenceType.Upper;

            if (type == SequenceType.Digit)
            {
                if (typeSeq == SequenceType.Unknown) typeSeq = type;
                else
                {
                    if (type != typeSeq) return false;
                }
            }
            else if (type == SequenceType.Upper)
            {
                if (typeSeq == SequenceType.Digit) return false;
                if (PanelSeg.isCaseSame(ch)) continue;

                if (typeSeq == SequenceType.Unknown) typeSeq = type;
                else
                {
                    if (type != typeSeq) return false;
                }
            }
            else if (type == SequenceType.Lower)
            {
                if (typeSeq == SequenceType.Digit) return false;
                if (PanelSeg.isCaseSame(ch)) continue;

                if (typeSeq == SequenceType.Unknown) typeSeq = type;
                else
                {
                    if (type != typeSeq) return false;
                }
            }
        }
        return true;
    }

    static float[] featureExtraction(opencv_core.Mat image, Panel[] panelSeq) {
        int n = panelSeq.length;

        //Collect various info from image and panelSeq
        int imageWidth = image.cols(), imageHeight = image.rows();

        double meanArea = 0.0; double meanSide = 0.0;
        opencv_core.Point2f[] centers = new opencv_core.Point2f[n];
        for (int i = 0; i < n; i++) {
            Panel panel = panelSeq[i];
            Rectangle labelRect = panel.labelRect;

            double area = labelRect.getHeight() * labelRect.getWidth();
            meanArea += area;

            meanSide += labelRect.getHeight(); //We have been using squares, so width == height.

            opencv_core.Point2f center = new opencv_core.Point2f((labelRect.x + labelRect.width) / 2.0f, (labelRect.y + labelRect.height) / 2.0f);
            centers[i] = center;
        }
        meanArea /= n;
        meanSide /= n;

        //Extract features
        ArrayList<Float> features = new ArrayList<>();

        //Alignment (we find the shortest horizontal or vertical distance (we want a good x- or y- alignment)
        //and the corresponding vertical or horizontal distance (but, we don't want they are very close on the other direction).
        for (int i = 0; i < n; i++)
        {
            opencv_core.Point2f center1 = centers[i];
            float minDis = Float.MAX_VALUE; float minDisPair = 0.0f;
            for (int j = 0; j < n; j++)
            {
                if (j == i) continue;
                opencv_core.Point2f center2 = centers[j];
                float xDis = Math.abs(center1.x()-center2.x()); //xDis/=meanSide;
                float yDis = Math.abs(center1.y()-center2.y()); //yDis/=meanSide;
                if (xDis < minDis)
                {
                    minDis = xDis; minDisPair = yDis;
                }
                if (yDis < minDis)
                {
                    minDis = yDis; minDisPair = xDis;
                }
            }
            features.add(minDis); features.add(minDisPair);
        }


        //Area ratio to mean area, number close to 1 indicates 
//        for (int i = 0; i < n;i++)
//        {
//            Panel panel = panelSeq.get(i);
//            double area = panel.labelRect.getHeight() * panel.labelRect.getWidth();
//            double feature = area / meanArea;
//            features.add(feature);
//        }

        Float[] fs = features.toArray(new Float[features.size()]);
        return ArrayUtils.toPrimitive(fs);

        //return features;
    }
}
