package gov.nih.nlm.lhc.openi.panelseg;

import akka.japi.pf.FI;
import org.bytedeco.javacpp.indexer.DoubleRawIndexer;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.IntRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_ximgproc;
import org.jetbrains.annotations.Nullable;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_core.CV_32FC1;
import static org.bytedeco.javacpp.opencv_core.CV_REDUCE_SUM;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_highgui.waitKey;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * Use Structured Edge to try split panels
 *
 * Created by jzou on 5/19/2017.
 */
public class PanelSplitStructuredEdge
{
    Figure figure;

    static opencv_ximgproc.StructuredEdgeDetection se;
    static opencv_imgproc.LineSegmentDetector lsd;

    List<Rect> separators;  //The separatorBands (vertical and horizontal uniform bands) or
                            // separatorLines (vertical and horizontal high gradient edges) detected.
    List<Rect[]> separatorRects;    //The rects split by the separators

    Mat edgeCCLabels;       //The binary edge connected components labels
    Mat edgeCCStats;        //The binary edge connected components statistics
    Mat edgeCCCentroids;    //The binary edge connected components centroids
    List<CCInfo> edgeConnectedComponents; //The connected components after some merging

    static void initialize() throws Exception
    {
        se = opencv_ximgproc.createStructuredEdgeDetection("/Users/jie/projects/PanelSeg/programs/PanelSegJ/models/model.yml.gz");
        lsd = createLineSegmentDetector();
    }

    PanelSplitStructuredEdge(Figure figure) {this.figure = figure;         separators = new ArrayList<>();}

    void detectEdges()
    {
        detectStructuredEdge();
        createBinaryEdgeMap();
    }

    void ccAnalysis()
    {
        collectEdgeConnectedComponents();
        mergeCompletelyEnclosedCCs();
    }

    private void detectStructuredEdge()
    {
        opencv_core.Mat normImg = new opencv_core.Mat();
        figure.imageOriginal.convertTo(normImg, opencv_core.CV_32F,  1.0 / 255.0, 0.0);

        opencv_core.Mat edge = new opencv_core.Mat();
        se.detectEdges(normImg, edge);

        //figure.structuredEdge = edge;
        figure.structuredEdge = new Mat();
        opencv_core.copyMakeBorder(edge, figure.structuredEdge, Figure.padding, Figure.padding, Figure.padding, Figure.padding, opencv_core.BORDER_CONSTANT, new opencv_core.Scalar());
    }

    void createBinaryEdgeMap()
    {
        int width = figure.structuredEdge.cols(), height= figure.structuredEdge.rows();
        opencv_core.Mat grayEdge = new opencv_core.Mat(height, width, CV_8UC1);
        figure.structuredEdge.convertTo(grayEdge, CV_8U, 255, 0);
        figure.binaryEdgeMap = new opencv_core.Mat(height, width, CV_8UC1);
        threshold(grayEdge, figure.binaryEdgeMap, 0, 255, THRESH_BINARY | THRESH_OTSU);
    }

    void collectEdgeConnectedComponents()
    {
        //Find and collect all CC's
        edgeCCLabels = new Mat(); edgeCCStats = new Mat();   edgeCCCentroids = new Mat();
        connectedComponentsWithStats(figure.binaryEdgeMap, edgeCCLabels, edgeCCStats, edgeCCCentroids);

        edgeConnectedComponents = new ArrayList<>();
        IntRawIndexer indexer = edgeCCStats.createIndexer();
        for (int i = 0; i < edgeCCStats.rows(); i++)
        {
            int left = indexer.get(i, CC_STAT_LEFT);
            int top = indexer.get(i, CC_STAT_TOP);
            int width = indexer.get(i, CC_STAT_WIDTH);
            int height = indexer.get(i, CC_STAT_HEIGHT);
            int size = indexer.get(i, CC_STAT_AREA );

            if (left == 0||top == 0 || width == edgeCCLabels.cols() || height == edgeCCLabels.rows()) continue; //ignore

            CCInfo cc = new CCInfo(left, top, width, height, size);
            edgeConnectedComponents.add(cc);
        }

    }

    void mergeCompletelyEnclosedCCs()
    {
        //Merge CC's which are completely inside a larger CC.
        List<CCInfo> ccsNew = new ArrayList<>();
        while (edgeConnectedComponents.size() > 0)
        {
            //Find the maximum CC
            CCInfo maxCC = edgeConnectedComponents.get(0);
            for (int i = 1; i < edgeConnectedComponents.size(); i++)
            {
                CCInfo cc = edgeConnectedComponents.get(i);
                if (cc.size > maxCC.size) maxCC = cc;
            }
            //Remove CCs in ccs, which are completely enclosed by maxCC
            for (int i = edgeConnectedComponents.size() - 1; i >= 0; i--)
            {
                CCInfo cc = edgeConnectedComponents.get(i);
                if (maxCC.rectangle.contains(cc.rectangle))
                {
                    edgeConnectedComponents.remove(i);
                    maxCC.size += cc.size;
                }
            }
            ccsNew.add(maxCC);
        }

        edgeConnectedComponents = ccsNew;
    }

    List<Panel> splitByCCAnalysis(Panel panel, List<Panel> labels)
    {
        Rect roi_orig = AlgOpenCVEx.Rectangle2Rect(panel.panelRect);

        //Before we do any analysis, we update its bounding box.
        opencv_core.Mat edges = new opencv_core.Mat(figure.binaryEdgeMap, roi_orig); //Work on ROI only
        opencv_core.Rect roi_inside = AlgOpenCVEx.findBoundingbox(edges);
        opencv_core.Rect roi = new opencv_core.Rect(roi_orig.x() + roi_inside.x(), roi_orig.y() + roi_inside.y(), roi_inside.width(), roi_inside.height());

        //Split by CC analysis
        // 1. Collect all qualified cc's in panel
        List<CCInfo> ccs = new ArrayList<>();
        for (CCInfo cc : edgeConnectedComponents)
        {
            Rectangle ccRect = cc.rectangle;
            if (ccRect.width < 100 || ccRect.height < 100) continue;
            if (panel.panelRect.contains(cc.rectangle)) ccs.add(cc);
        }
        // 2. For each label, find one qualified cc to form a qualified panel. If not able to find, remove that label
        List<Panel> panelsNew = new ArrayList<>();
        for (Panel label : labels)
        {
            //find max overlapping
            int maxIndex = AlgMiscEx.maxOverlappingCC(label.labelRect, ccs);
            if (maxIndex != -1)
            {
                label.panelRect = ccs.get(maxIndex).rectangle;
                panelsNew.add(label);
                ccs.remove(maxIndex);
                continue;
            }

            //No overlapping, we find the closest panel
            int minIndex = AlgMiscEx.closestCC(label.labelRect, ccs);
            if (minIndex != -1)
            {
                label.panelRect = ccs.get(minIndex).rectangle;
                panelsNew.add(label);
                ccs.remove(minIndex);
                continue;
            }
            //Reach here, no CC matches to the label, remove the label
        }

        // 3. For the remaining cc, from max to min, merge them into the closet qualified panel
        for (CCInfo cc : edgeConnectedComponents)
        {
            if (!panel.panelRect.contains(cc.rectangle)) continue;

            Rectangle ccRect = cc.rectangle;

            //find max overlapping
            int maxIndex = AlgMiscEx.maxOverlappingPanel(ccRect, panelsNew);
            if (maxIndex != -1)
            {
                Panel panelNew = panelsNew.get(maxIndex);
                Rectangle rectangle = panelNew.panelRect.union(ccRect);
                panelsNew.get(maxIndex).panelRect = rectangle;
                continue;
            }

            //No overlapping, we find the closest panel
            int minIndex = AlgMiscEx.closestPanel(ccRect, panelsNew);
            if (minIndex != -1)
            {
                Panel panelNew = panelsNew.get(minIndex);
                Rectangle rectangle = panelNew.panelRect.union(ccRect);
                panelsNew.get(minIndex).panelRect = rectangle;
            }
        }

        return panelsNew;
    }

    List<Panel> splitByBandAndLine(Panel panel)
    {
        Rect roi_orig = AlgOpenCVEx.Rectangle2Rect(panel.panelRect);

        //Before we do any analysis, we update its bounding box.
        opencv_core.Mat edges = new opencv_core.Mat(figure.binaryEdgeMap, roi_orig); //Work on ROI only
        opencv_core.Rect roi_inside = AlgOpenCVEx.findBoundingbox(edges);
        opencv_core.Rect roi = new opencv_core.Rect(roi_orig.x() + roi_inside.x(), roi_orig.y() + roi_inside.y(), roi_inside.width(), roi_inside.height());

        opencv_core.Rect verBand = verticalCutByBand(roi, 60);
        opencv_core.Rect horBand = horizontalCutByBand(roi, 60);
        opencv_core.Rect verLine = verticalCutByLine(roi);
        opencv_core.Rect horLine = horizontalCutByLine(roi);

        Rect[] rois = null;
        if (verBand != null || horBand != null) {   //Cut by bands
            boolean toCutVertical = true;
            if (verBand == null && horBand != null) toCutVertical = false;
            else if (verBand != null && horBand != null) {
                if (verBand.width() < horBand.height()) toCutVertical = false;
            }

            if (toCutVertical) {
                //Recover to original roi
                verBand = new opencv_core.Rect(verBand.x(), roi_orig.y(), verBand.width(), roi_orig.height());
                separators.add(verBand);

                //Break into 2 zones vertically
                int left, top, right, bottom;
                left = roi_orig.x();
                top = roi_orig.y();
                right = verBand.x() + verBand.width() / 2;
                bottom = top + roi_orig.height();
                opencv_core.Rect roi1 = new opencv_core.Rect(left, top, right - left, bottom - top);

                left = verBand.x() + verBand.width() / 2;
                top = roi_orig.y();
                right = roi_orig.x() + roi_orig.width();
                bottom = top + roi_orig.height();
                opencv_core.Rect roi2 = new opencv_core.Rect(left, top, right - left, bottom - top);

                rois = new Rect[2];
                rois[0] = roi1;
                rois[1] = roi2;
//                separatorRects.add(rois);
//                split(roi1);                split(roi2);
            } else {
                //Recover to original roi
                horBand = new opencv_core.Rect(roi_orig.x(), horBand.y(), roi_orig.width(), horBand.height());
                separators.add(horBand);

                //Break into 2 zones horizontally
                int left, top, right, bottom;
                left = roi_orig.x();
                top = roi_orig.y();
                right = roi_orig.x() + roi.width();
                bottom = horBand.y() + horBand.height() / 2;
                opencv_core.Rect roi1 = new opencv_core.Rect(left, top, right - left, bottom - top);

                left = roi_orig.x();
                top = horBand.y() + horBand.height() / 2;
                right = roi_orig.x() + roi_orig.width();
                bottom = roi_orig.y() + roi_orig.height();
                opencv_core.Rect roi2 = new opencv_core.Rect(left, top, right - left, bottom - top);

                rois = new Rect[2];
                rois[0] = roi1;
                rois[1] = roi2;
//                separatorRects.add(rois);
//                split(roi1);                split(roi2);
            }
        } else if (verLine != null || horLine != null) {   //Cut by lines
            boolean toCutVertical = true;
            if (verLine == null && horLine != null) toCutVertical = false;
            else if (verLine != null && horLine != null) {
                if (verLine.width() < horLine.height()) toCutVertical = false;
            }

            if (toCutVertical) {
                //Recover to original roi
                verLine = new opencv_core.Rect(verLine.x(), roi_orig.y(), verLine.width(), roi_orig.height());
                separators.add(verLine);

                //Break into 2 zones vertically
                int left, top, right, bottom;
                left = roi_orig.x();
                top = roi_orig.y();
                right = verLine.x() + verLine.width() / 2;
                bottom = top + roi_orig.height();
                opencv_core.Rect roi1 = new opencv_core.Rect(left, top, right - left, bottom - top);

                left = verLine.x() + verLine.width() / 2;
                top = roi_orig.y();
                right = roi_orig.x() + roi_orig.width();
                bottom = top + roi_orig.height();
                opencv_core.Rect roi2 = new opencv_core.Rect(left, top, right - left, bottom - top);

                rois = new Rect[2];
                rois[0] = roi1;
                rois[1] = roi2;
//                separatorRects.add(rois);
//                split(roi1);                split(roi2);
            } else {
                //Recover to original roi
                horLine = new opencv_core.Rect(roi_orig.x(), horLine.y(), roi_orig.width(), horLine.height());
                separators.add(horLine);

                //Break into 2 zones horizontally
                int left, top, right, bottom;
                left = roi_orig.x();
                top = roi_orig.y();
                right = roi_orig.x() + roi.width();
                bottom = horLine.y() + horLine.height() / 2;
                opencv_core.Rect roi1 = new opencv_core.Rect(left, top, right - left, bottom - top);

                left = roi_orig.x();
                top = horLine.y() + horLine.height() / 2;
                right = roi_orig.x() + roi_orig.width();
                bottom = roi_orig.y() + roi_orig.height();
                opencv_core.Rect roi2 = new opencv_core.Rect(left, top, right - left, bottom - top);

                rois = new Rect[2];
                rois[0] = roi1;
                rois[1] = roi2;
//                separatorRects.add(rois);
//                split(roi1);                split(roi2);
            }
        }

        List<Panel> panels = new ArrayList<>();
        if (rois != null)
        {
            for (int i = 0; i < rois.length; i++)
            {
                Rectangle rectangle = new Rectangle(rois[i].x(), rois[i].y(), rois[i].width(), rois[i].height());
                Panel panelT = new Panel(); panelT.panelRect = rectangle;
                panels.add(panelT);
            }
        }

        return panels;
    }

    List<Panel> splitByLabels(Panel panel, List<Panel> labels)
    {
        List<Panel> panels = new ArrayList<>();

        //Check whether labels are indeed very likely to be correct
        //1. Every label score needs to be above certain threshold
        for (Panel label : labels)
        {
            if (label.labelScore < 0.8) return panels;
        }
        //2. Every labels need to be consecutive
        labels.sort(new PanelLabelAscending());
        for (int i = 1; i < labels.size(); i++)
        {
            int label0 = (int)labels.get(i-1).panelLabel.toLowerCase().charAt(0);
            int label1 = (int)labels.get(i).panelLabel.toLowerCase().charAt(0);

            if (label1 - label0 != 1) return panels;
        }
        //3. Every label should have aligned to at least another one
        for (int i = 0; i < labels.size(); i++)
        {
            Panel label0 = labels.get(i); Rectangle rect0 = label0.labelRect;
            Boolean alignedFound = false;
            for (int j = 0; j < labels.size(); j++)
            {
                if (i == j) continue;
                Panel label1 = labels.get(j); Rectangle rect1 = label1.labelRect;
                if (AlgMiscEx.alignedRects(rect0, rect1))
                {
                    alignedFound = true; break;
                }
            }
            if (!alignedFound) return panels;
        }

        //Project edges horizontally or vertically
        Rect roi = new Rect(Figure.padding, Figure.padding, figure.imageOriginalWidth, figure.imageOriginalHeight);
        Mat edges = new Mat(figure.binaryEdgeMap, roi); //Work on ROI only
        int height = edges.rows(), width = edges.cols();

        Mat verProfile = new Mat(height, 1, CV_32FC1);
        reduce(edges, verProfile, 1, CV_REDUCE_SUM, CV_32FC1); //Horizontal projection generate vertical profile
        FloatRawIndexer verProfileIndex = verProfile.createIndexer();
        Mat horProfile = new Mat(1, width, CV_32FC1);
        reduce(edges, horProfile, 0, CV_REDUCE_SUM, CV_32FC1); //vertical projection generate horizontal profile
        FloatRawIndexer horProfileIndex = horProfile.createIndexer();

        //Reach here: labels are very likely to correct. We break the panels according to the location of labels.
        labels.sort(new LabelRectRowFirst());
        for (int i = 0; i < labels.size(); i++)
        {
            Panel label = labels.get(i);
            //Find left, top, right, bottom adjacent labels
            Panel adjLeft = null, adjTop = null, adjRight = null, adjBottom = null;
            int disLeft = 0, disTop = 0, disRight = 0, disBottom = 0;
            for (int j = 0; j < labels.size(); j++)
            {
                if (i == j) continue;

                Panel adjLabel = labels.get(j);
                //Check left
                if (adjLabel.labelRect.x + adjLabel.labelRect.width < label.labelRect.x)
                {
                    int dis = label.labelRect.x - (adjLabel.labelRect.x + adjLabel.labelRect.width);
                    if (adjLeft == null || dis < disLeft)
                    {
                        adjLeft = adjLabel; disLeft = dis;
                    }
                }
                //Check right
                if (adjLabel.labelRect.x > label.labelRect.x + label.labelRect.width)
                {
                    int dis = adjLabel.labelRect.x  - (label.labelRect.x + label.labelRect.width);
                    if (adjRight == null || dis < disRight)
                    {
                        adjRight = adjLabel; disRight = dis;
                    }
                }
                //Check top
                if (adjLabel.labelRect.y + adjLabel.labelRect.height < label.labelRect.y)
                {
                    int dis = label.labelRect.y - (adjLabel.labelRect.y + adjLabel.labelRect.height);
                    if (adjTop == null || dis < disTop)
                    {
                        adjTop = adjLabel; disTop = dis;
                    }
                }
                //Check bottom
                if (adjLabel.labelRect.y > label.labelRect.y + label.labelRect.height)
                {
                    int dis = adjLabel.labelRect.y  - (label.labelRect.y + label.labelRect.height);
                    if (adjBottom == null || dis < disBottom)
                    {
                        adjBottom = adjLabel; disBottom = dis;
                    }
                }
            }

            //Found left, top, right, bottom adjacent labels
            int left = 0, right = 0, top = 0, bottom = 0;
            if (adjLeft == null) left = panel.panelRect.x;
            else
            {
                int start = adjLeft.labelRect.x + adjLeft.labelRect.width;
                int end = label.labelRect.x;
                //start += (end-start)/3; end -= (end-start)/3; //We search in the middle 1/3 only

                //Try find the uniform band
                Rect roiBand = new Rect(start, panel.panelRect.y, end - start, panel.panelRect.height);
                opencv_core.Rect verBand = verticalCutByBand(roiBand, 0);
                if (verBand !=null)
                    left = verBand.x() + verBand.width()/2;
                else
                {   //If not find the maximum edge
                    float max = -1.0f;
                    for (int k = start; k < end; k++)
                    {
                        float f = horProfileIndex.get(0, k - Figure.padding);
                        if (f > max) { left = k; max = f;}
                    }
                }
            }
            if (adjRight == null) right = panel.panelRect.x + panel.panelRect.width;
            else
            {
                int start = label.labelRect.x + label.labelRect.width;
                int end = adjRight.labelRect.x;
                //start += (end-start)/3; end -= (end-start)/3; //We search in the middle 1/3 only

                //Try find the uniform band
                Rect roiBand = new Rect(start, panel.panelRect.y, end - start, panel.panelRect.height);
                opencv_core.Rect verBand = verticalCutByBand(roiBand, 0);
                if (verBand !=null)
                    right = verBand.x() + verBand.width()/2;
                else
                {
                    float max = -1.0f;
                    for (int k = start + 3; k < end - 3; k++)
                    {
                        float f = horProfileIndex.get(0, k - Figure.padding);
                        if (f > max) { right = k; max = f;}
                    }
                }
            }
            if (adjTop == null) top = panel.panelRect.y;
            else
            {
                int start = adjTop.labelRect.y + adjTop.labelRect.height;
                int end = label.labelRect.y;
                //start += (end-start)/3; end -= (end-start)/3; //We search in the middle 1/3 only

                //Try find the uniform band
                Rect roiBand = new Rect(panel.panelRect.x, start, panel.panelRect.width, end - start);
                opencv_core.Rect horBand = horizontalCutByBand(roiBand, 0);
                if (horBand !=null)
                    top = horBand.x() + horBand.width()/2;
                else
                {
                    float max = -1.0f;
                    for (int k = start + 3; k < end - 3; k++)
                    {
                        float f = verProfileIndex.get(0, k - Figure.padding);
                        if (f > max) { top = k; max = f;}
                    }
                }
            }
            if (adjBottom == null) bottom = panel.panelRect.y + panel.panelRect.height;
            else
            {
                int start = label.labelRect.y + label.labelRect.height;
                int end = adjBottom.labelRect.y;
                //start += (end-start)/3; end -= (end-start)/3; //We search in the middle 1/3 only

                //Try find the uniform band
                Rect roiBand = new Rect(panel.panelRect.x, start, panel.panelRect.width, end - start);
                opencv_core.Rect horBand = horizontalCutByBand(roiBand, 0);
                if (horBand !=null)
                    bottom = horBand.x() + horBand.width()/2;
                else
                {
                    float max = -1.0f;
                    for (int k = start + 3; k < end - 3; k++)
                    {
                        float f = verProfileIndex.get(0, k - Figure.padding);
                        if (f > max) { bottom = k; max = f;}
                    }
                }
            }

            label.panelRect = new Rectangle(left, top, right-left, bottom-top);
            panels.add(label);
        }

        return panels;
    }

    void split(Rectangle roi)
    {
        //Clears out separators
        separators = new ArrayList<>();
        separatorRects = new ArrayList<>();

        //Do recursive splits
        split(AlgOpenCVEx.Rectangle2Rect(roi));

        //Display splits for debugging purposes
        if (separators != null && separators.size() > 0)
        {
            //Draw separators
            opencv_core.Mat img = figure.imageColor.clone();
            for (int i = 0; i < separators.size(); i++)
            {
                opencv_core.Rect rect = separators.get(i);
                rectangle(img, rect, new opencv_core.Scalar(0, 255, 0, 0), CV_FILLED, 8, 0);
            }
            imshow("Separators", img);
            //waitKey();
        }
    }

    private void split(opencv_core.Rect roi_orig)
    {
        //Before we do any analysis, we update its bounding box.
        opencv_core.Mat edges = new opencv_core.Mat(figure.binaryEdgeMap, roi_orig); //Work on ROI only
        opencv_core.Rect roi_inside = AlgOpenCVEx.findBoundingbox(edges);
        opencv_core.Rect roi = new opencv_core.Rect(roi_orig.x()+roi_inside.x(), roi_orig.y() + roi_inside.y(), roi_inside.width(), roi_inside.height());

        opencv_core.Rect verBand = verticalCutByBand(roi, 60);
        opencv_core.Rect horBand = horizontalCutByBand(roi, 60);
        opencv_core.Rect verLine = verticalCutByLine(roi);
        opencv_core.Rect horLine = horizontalCutByLine(roi);

        Rect[] rois = null;
        if (verBand != null || horBand != null)
        {   //Cut by bands
            boolean toCutVertical = true;
            if (verBand == null && horBand !=null) toCutVertical = false;
            else if (verBand != null && horBand != null)
            {
                if (verBand.width() < horBand.height()) toCutVertical = false;
            }

            if (toCutVertical)
            {
                //Recover to original roi
                verBand = new opencv_core.Rect(verBand.x(), roi_orig.y(), verBand.width(), roi_orig.height());
                separators.add(verBand);

                //Break into 2 zones vertically
                int left, top, right, bottom;
                left = roi_orig.x(); top = roi_orig.y(); right = verBand.x()+verBand.width()/2; bottom = top + roi_orig.height();
                opencv_core.Rect roi1 = new opencv_core.Rect(left, top, right-left, bottom-top);

                left = verBand.x()+verBand.width()/2; top = roi_orig.y(); right = roi_orig.x()+roi_orig.width(); bottom = top + roi_orig.height();
                opencv_core.Rect roi2 = new opencv_core.Rect(left, top, right-left, bottom-top);

                rois = new Rect[2]; rois[0] = roi1; rois[1] = roi2;
                separatorRects.add(rois);
                split(roi1);                split(roi2);
            }
            else
            {
                //Recover to original roi
                horBand = new opencv_core.Rect(roi_orig.x(), horBand.y(), roi_orig.width(), horBand.height());
                separators.add(horBand);

                //Break into 2 zones horizontally
                int left, top, right, bottom;
                left = roi_orig.x(); top = roi_orig.y(); right = roi_orig.x() + roi.width(); bottom = horBand.y()+horBand.height()/2;
                opencv_core.Rect roi1 = new opencv_core.Rect(left, top, right-left, bottom-top);

                left = roi_orig.x(); top = horBand.y()+horBand.height()/2; right = roi_orig.x()+roi_orig.width(); bottom = roi_orig.y() + roi_orig.height();
                opencv_core.Rect roi2 = new opencv_core.Rect(left, top, right-left, bottom-top);

                rois = new Rect[2]; rois[0] = roi1; rois[1] = roi2;
                separatorRects.add(rois);
                split(roi1);                split(roi2);
            }
        }
        else if (verLine != null || horLine != null)
        {   //Cut by lines
            boolean toCutVertical = true;
            if (verLine == null && horLine != null) toCutVertical = false;
            else if (verLine != null && horLine != null)
            {
                if (verLine.width() < horLine.height()) toCutVertical = false;
            }

            if (toCutVertical)
            {
                //Recover to original roi
                verLine = new opencv_core.Rect(verLine.x(), roi_orig.y(), verLine.width(), roi_orig.height());
                separators.add(verLine);

                //Break into 2 zones vertically
                int left, top, right, bottom;
                left = roi_orig.x(); top = roi_orig.y(); right = verLine.x()+verLine.width()/2; bottom = top + roi_orig.height();
                opencv_core.Rect roi1 = new opencv_core.Rect(left, top, right-left, bottom-top);

                left = verLine.x()+verLine.width()/2; top = roi_orig.y(); right = roi_orig.x()+roi_orig.width(); bottom = top + roi_orig.height();
                opencv_core.Rect roi2 = new opencv_core.Rect(left, top, right-left, bottom-top);

                rois = new Rect[2]; rois[0] = roi1; rois[1] = roi2;
                separatorRects.add(rois);
                split(roi1);                split(roi2);
            }
            else
            {
                //Recover to original roi
                horLine = new opencv_core.Rect(roi_orig.x(), horLine.y(), roi_orig.width(), horLine.height());
                separators.add(horLine);

                //Break into 2 zones horizontally
                int left, top, right, bottom;
                left = roi_orig.x(); top = roi_orig.y(); right = roi_orig.x() + roi.width(); bottom = horLine.y()+horLine.height()/2;
                opencv_core.Rect roi1 = new opencv_core.Rect(left, top, right-left, bottom-top);

                left = roi_orig.x(); top = horLine.y()+horLine.height()/2; right = roi_orig.x()+roi_orig.width(); bottom = roi_orig.y() + roi_orig.height();
                opencv_core.Rect roi2 = new opencv_core.Rect(left, top, right-left, bottom-top);

                rois = new Rect[2]; rois[0] = roi1; rois[1] = roi2;
                separatorRects.add(rois);
                split(roi1);                split(roi2);
            }
        }
    }

    @Nullable
    private opencv_core.Rect horizontalCutByBand(opencv_core.Rect roi, int dis_to_boundary_ignore)
    {
        List<opencv_core.Rect> bands = collectHorizontalBands(roi, dis_to_boundary_ignore);

        if (bands.size() > 0)
        {
            //We find the maximum width
            opencv_core.Rect maxRect = bands.get(0);
            for (int i = 1; i < bands.size(); i++)
            {
                opencv_core.Rect rect = bands.get(i);
                if (rect.height() > maxRect.height())
                    maxRect = rect;
            }
            return maxRect;
        }

        return null;
    }

    private List<opencv_core.Rect> collectHorizontalBands(opencv_core.Rect roi, int dis_to_boundary_ignore)
    {
        opencv_core.Mat edges = new opencv_core.Mat(figure.binaryEdgeMap, roi); //Work on ROI only
        int height = edges.rows();

        //horizontal projection
        opencv_core.Mat verProfile = new opencv_core.Mat(height, 1, CV_32FC1);
        reduce(edges, verProfile, 1, CV_REDUCE_SUM, CV_32FC1); //Horizontal projection generate vertical profile

        int[] profile = new int[height];
        FloatRawIndexer verProfileIndex = verProfile.createIndexer();
        for (int i = 0; i < height; i++)
        {
            float f = verProfileIndex.get(0, i);
            profile[i] = (int)(f/255);
        }

        //Find all 0 runs (no edge separatorBands)
        //Calculate Gradients
        int[] grads = new int[height];
        for (int i = 1; i < height; i++)
        {
            int p1 = profile[i] == 0 ? 0 : (profile[i] < 0 ? -1 : 1);
            int p0 = profile[i-1] == 0 ? 0 : (profile[i-1] < 0 ? -1 : 1);
            grads[i] = p1 - p0;
        }

        List<Integer> tops = new ArrayList<>(); List<Integer> bottoms = new ArrayList<>();
        for (int i = 1; i < height; i++)
        {
            int grad = grads[i];
            if (grad < 0)
            {
                int top = i; int j;
                for (j = i+1; j <height; j++)
                {
                    grad = grads[j];
                    if (grad > 0 || (dis_to_boundary_ignore == 0 && j == height - 1)) //when we do not want to ignore boundary bands,
                                                                                      //We set dis_to_boundary_ignore to be 0
                    {
                        int bottom = j;
                        tops.add(top); bottoms.add(bottom);
                        break;
                    }
                }
                i = j;
            }
        }

        List<Rect> rects = new ArrayList<>();
        for (int i = 0; i < tops.size(); i++)
        {
            int top = tops.get(i), bottom = bottoms.get(i);

            //Ignore some simple cases
            if (bottom - top < 5) continue; //If the band is too narrow, we ignore
            if (top < dis_to_boundary_ignore || height - bottom < dis_to_boundary_ignore) continue; //If the band is too close to the top and bottom boundary, we ignore

            Rect rect = new Rect(roi.x(), roi.y()+ top, roi.width(), bottom - top);

            if (!uniformBand(rect)) continue; //If on grayscale image, it is not uniform, we ignore.

            rects.add(rect);
        }

        return rects;
    }

    @Nullable
    Rect verticalCutByBand(Rect roi, int dis_to_boundary_ignore)
    {
        List<Rect> bands = collectVerticalBands(roi, dis_to_boundary_ignore);

        if (bands.size() > 0)
        {
            //We find the maximum width
            Rect maxRect = bands.get(0);
            for (int i = 1; i < bands.size(); i++)
            {
                Rect rect = bands.get(i);
                if (rect.width() > maxRect.width())
                    maxRect = rect;
            }
            return maxRect;
        }

        return null;
    }

    List<Rect> collectVerticalBands(Rect roi, int dis_to_boundary_ignore)
    {
        Mat edges = new Mat(figure.binaryEdgeMap, roi); //Work on ROI only
//        imshow("edges", edges);
//        waitKey();
        int width = edges.cols();

        //vertical projection
        Mat horProfile = new Mat(1, width, CV_32FC1);
        reduce(edges, horProfile, 0, CV_REDUCE_SUM, CV_32FC1); //Vertical projection generate Horizontal profile

        int[] profile = new int[width];
        FloatRawIndexer horProfileIndex = horProfile.createIndexer();
        for (int i = 0; i < width; i++)
        {
            float f = horProfileIndex.get(0, i);
            profile[i] = (int)(f/255);
        }

        //Find all 0 runs (no edge separatorBands)
        //Calculate Gradients
        int[] grads = new int[width];
        for (int i = 1; i < width; i++)
        {
            int p1 = profile[i] == 0 ? 0 : (profile[i] < 0 ? -1 : 1);
            int p0 = profile[i-1] == 0 ? 0 : (profile[i-1] < 0 ? -1 : 1);
            grads[i] = p1 - p0;
        }

        List<Integer> lefts = new ArrayList<>(); List<Integer> rights = new ArrayList<>();
        for (int i = 1; i < width; i++)
        {
            int grad = grads[i];
            if (grad < 0)
            {
                int left = i; int j;
                for (j = i+1; j <width; j++)
                {
                    grad = grads[j];
                    if (grad > 0 || (dis_to_boundary_ignore == 0 && j == width - 1)) //when we do not want to ignore boundary bands,
                                                                                     //We set dis_to_boundary_ignore to be 0
                    {
                        int right = j;
                        lefts.add(left); rights.add(right);
                        break;
                    }
                }
                i = j;
            }
        }

        List<Rect> rects = new ArrayList<>();
        for (int i = 0; i < lefts.size(); i++)
        {
            int left = lefts.get(i), right = rights.get(i);

            //Ignore some simple cases
            if (right - left < 5) continue; //If the band is too narrow, we ignore
            if (left < dis_to_boundary_ignore || width - right < dis_to_boundary_ignore) continue; //If the band is too close to the left and right boundary, we ignore

            Rect rect = new Rect(roi.x() + left, roi.y(), right - left, roi.height());

            if (!uniformBand(rect)) continue; //If on grayscale image, it is not uniform, we ignore.

            rects.add(rect);
        }

        return rects;
    }

    boolean uniformBand(Rect rect)
    {
        Mat img = new Mat(figure.imageGray, rect);
        Mat meanMat = new Mat(); Mat stddevMat = new Mat();
        meanStdDev(img, meanMat, stddevMat);

        DoubleRawIndexer meanIndexer = meanMat.createIndexer();
        DoubleRawIndexer stddevIndexer = stddevMat.createIndexer();

        //double mean = meanIndexer.get(0);
        double stddev = stddevIndexer.get(0);

        if (stddev > 15.0) return false;

        return true;
    }

    @Nullable
    Rect horizontalCutByLine(Rect roi)
    {
        List<Rect> lines = collectHorizontalLines(roi);

        if (lines.size() > 0)
        {
            //We find the maximum width
            Rect maxRect = lines.get(0);
            for (int i = 1; i < lines.size(); i++)
            {
                Rect rect = lines.get(i);
                if (rect.height() > maxRect.height())
                    maxRect = rect;
            }
            return maxRect;
        }

        return null;
    }

    List<Rect> collectHorizontalLines(Rect roi)
    {
        Mat edges = new Mat(figure.binaryEdgeMap, roi); //Work on ROI only
        int height = edges.rows(), width = edges.cols();

        //horizontal projection
        Mat verProfile = new Mat(height, 1, CV_32FC1);
        reduce(edges, verProfile, 1, CV_REDUCE_SUM, CV_32FC1); //Horizontal projection generate vertical profile

        //Find all >0.9 runs (high gradient separatorLines)
        int[] profile = new int[height];
        FloatRawIndexer verProfileIndex = verProfile.createIndexer();
        for (int i = 0; i < height; i++)
        {
            float f = verProfileIndex.get(0, i);
            profile[i] = (f/255)/width > 0.8 ? 1 : 0;
        }
        //Calculate Gradients
        int[] grads = new int[height];
        for (int i = 1; i < height; i++)
        {
            int p1 = profile[i], p0 = profile[i-1];
            grads[i] = p1 - p0;
        }
        List<Integer> tops = new ArrayList<>(); List<Integer> bottoms = new ArrayList<>();
        for (int i = 1; i < height; i++)
        {
            int grad = grads[i];
            if (grad > 0)
            {
                int top = i; int j;
                for (j = i+1; j <height; j++)
                {
                    grad = grads[j];
                    if (grad < 0)
                    {
                        int bottom = j;
                        tops.add(top); bottoms.add(bottom);
                        break;
                    }
                }
                i = j;
            }
        }

        List<Rect> rects = new ArrayList<>();
        for (int i = 0; i < tops.size(); i++)
        {
            int top = tops.get(i), bottom = bottoms.get(i);

            //Ignore some simple cases
            if (top < 60 || height - bottom < 60) continue; //If the line is too close to the top and bottom boundary, we ignore

            Rect rect = new Rect(roi.x(), roi.y()+ top, roi.width(), bottom - top);
            rects.add(rect);
        }

        return rects;
    }

    @Nullable
    Rect verticalCutByLine(Rect roi)
    {
        List<Rect> lines = collectVerticalLines(roi);

        if (lines.size() > 0)
        {
            //We find the maximum height
            Rect maxRect = lines.get(0);
            for (int i = 1; i < lines.size(); i++)
            {
                Rect rect = lines.get(i);
                if (rect.width() > maxRect.width())
                    maxRect = rect;
            }
            return maxRect;
        }

        return null;
    }

    List<Rect> collectVerticalLines(Rect roi)
    {
        Mat edges = new Mat(figure.binaryEdgeMap, roi); //Work on ROI only
        int height = edges.rows(), width = edges.cols();

        //vertical projection
        Mat horProfile = new Mat(1, width, CV_32FC1);
        reduce(edges, horProfile, 0, CV_REDUCE_SUM, CV_32FC1); //vertical projection generate horizontal profile

        //Find all >0.8 runs (high gradient separatorLines)
        int[] profile = new int[width];
        FloatRawIndexer horProfileIndex = horProfile.createIndexer();
        for (int i = 0; i < width; i++)
        {
            float f = horProfileIndex.get(0, i);
            profile[i] = (f/255)/height > 0.8 ? 1 : 0;
        }
        //Calculate Gradients
        int[] grads = new int[width];
        for (int i = 1; i < width; i++)
        {
            int p1 = profile[i], p0 = profile[i-1];
            grads[i] = p1 - p0;
        }
        List<Integer> lefts = new ArrayList<>(); List<Integer> rights = new ArrayList<>();
        for (int i = 1; i < width; i++)
        {
            int grad = grads[i];
            if (grad > 0)
            {
                int left = i; int j;
                for (j = i+1; j <width; j++)
                {
                    grad = grads[j];
                    if (grad < 0)
                    {
                        int right = j;
                        lefts.add(left); rights.add(right);
                        break;
                    }
                }
                i = j;
            }
        }

        List<Rect> rects = new ArrayList<>();
        for (int i = 0; i < lefts.size(); i++)
        {
            int left = lefts.get(i), right = rights.get(i);

            //Ignore some simple cases
            if (left < 60 || width - right < 60) continue; //If the line is too close to the left and right boundary, we ignore

            Rect rect = new Rect(roi.x() + left, roi.y(), right - left, roi.height());
            rects.add(rect);
        }
        return rects;
    }
}
