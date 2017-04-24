package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.indexer.DoubleRawIndexer;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.IntRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_ximgproc;
import org.jetbrains.annotations.Nullable;

import java.util.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_highgui.waitKey;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgproc.THRESH_BINARY_INV;

/**
 * Panel Split method based on Edge Boxes
 *
 * Created by jzou on 2/6/2017.
 */
final class PanelSplitEdgeBox
{
    Figure figure;
    static opencv_ximgproc.StructuredEdgeDetection se;
    static opencv_imgproc.LineSegmentDetector lsd;

    static void initialize()
    {
        se = opencv_ximgproc.createStructuredEdgeDetection("/Users/jie/projects/PanelSeg/programs/PanelSegJ/models/model.yml.gz");
        lsd = createLineSegmentDetector();
    }

    PanelSplitEdgeBox(Figure figure) {this.figure = figure; }

    Mat structuredEdge;     //The edges detected by StructuredEdgeDetection
    Mat binaryEdgeMap;      //The binary edge map after Otsu thresholding

    Mat edgeCCLabels;       //The binary edge connected components labels
    Mat edgeCCStats;        //The binary edge connected components statistics
    Mat edgeCCCentroids;    //The binary edge connected components centroids

    List<Rect> separatorBands;   //The separatorBands (vertical and horizontal uniform bands or high gradient lines) detected.
    List<CCInfo> edgeConnectedComponents; //The connected components after some merging

    void split()
    {
        imshow("Image", figure.imageColor);
        detectStructuredEdge();
        imshow("StructuredEdges", structuredEdge);
        createBinaryEdgeMap();
        imshow("Binary Edges", binaryEdgeMap);

        splitByEdgeMap();

        collectEdgeConnectedComponents();
        mergeCompletelyEnclosedCCs();

        displayPanelCandidates();

        //detectLineSegments();
        waitKey();
    }

    void detectStructuredEdge()
    {
        Mat normImg = new Mat();
        figure.imageOriginal.convertTo(normImg, opencv_core.CV_32F,  1.0 / 255.0, 0.0);

        Mat edge = new Mat();
        se.detectEdges(normImg, edge);

        structuredEdge = edge;
//        structuredEdge = new Mat();
//        opencv_core.copyMakeBorder(edge, structuredEdge, Figure.padding, Figure.padding, Figure.padding, Figure.padding, opencv_core.BORDER_CONSTANT, new opencv_core.Scalar());
    }

    void createBinaryEdgeMap()
    {
        int width = structuredEdge.cols(), height= structuredEdge.rows();
        opencv_core.Mat grayEdge = new opencv_core.Mat(height, width, CV_8UC1);
        structuredEdge.convertTo(grayEdge, CV_8U, 255, 0);
        binaryEdgeMap = new opencv_core.Mat(height, width, CV_8UC1);
        threshold(grayEdge, binaryEdgeMap, 0, 255, THRESH_BINARY | THRESH_OTSU);
    }

    void splitByEdgeMap()
    {
        Rect roi = new Rect(0, 0, binaryEdgeMap.cols(), binaryEdgeMap.rows());

        separatorBands = new ArrayList<>();
        splitByEdgeMap(roi);
    }

    void splitByEdgeMap(Rect roi_orig)
    {
        //Before we do any analysis, we update its bounding box.
        Mat edges = new Mat(binaryEdgeMap, roi_orig); //Work on ROI only
        Rect roi_inside = AlgOpenCVEx.findBoundingbox(edges);
        Rect roi = new Rect(roi_orig.x()+roi_inside.x(), roi_orig.y() + roi_inside.y(), roi_inside.width(), roi_inside.height());

        Rect verCut = verticalCut(roi);
        Rect horCut = horizontalCut(roi);

        if (verCut == null && horCut == null) return;

        boolean toCutVertical = true;
        if (verCut == null && horCut !=null) toCutVertical = false;
        else if (verCut != null && horCut != null)
        {
            if (verCut.width() < horCut.height()) toCutVertical = false;
        }

        if (toCutVertical)
        {
            //Recover to original roi
            verCut = new Rect(verCut.x(), roi_orig.y(), verCut.width(), roi_orig.height());
            separatorBands.add(verCut);

            //Break into 2 zones vertically
            int left, top, right, bottom;
            left = roi_orig.x(); top = roi_orig.y(); right = verCut.x()+verCut.width()/2; bottom = top + roi_orig.height();
            Rect roi1 = new Rect(left, top, right-left, bottom-top);
            splitByEdgeMap(roi1);

            left = verCut.x()+verCut.width()/2; top = roi_orig.y(); right = roi_orig.x()+roi_orig.width(); bottom = top + roi_orig.height();
            Rect roi2 = new Rect(left, top, right-left, bottom-top);
            splitByEdgeMap(roi2);
        }
        else
        {
            //Recover to original roi
            horCut = new Rect(roi_orig.x(), horCut.y(), roi_orig.width(), horCut.height());
            separatorBands.add(horCut);

            //Break into 2 zones horizontally
            int left, top, right, bottom;
            left = roi_orig.x(); top = roi_orig.y(); right = roi_orig.x() + roi.width(); bottom = horCut.y()+horCut.height()/2;
            Rect roi1 = new Rect(left, top, right-left, bottom-top);
            splitByEdgeMap(roi1);

            left = roi_orig.x(); top = horCut.y()+horCut.height()/2; right = roi_orig.x()+roi_orig.width(); bottom = roi_orig.y() + roi_orig.height();
            Rect roi2 = new Rect(left, top, right-left, bottom-top);
            splitByEdgeMap(roi2);
        }
    }

    @Nullable
    Rect horizontalCut(Rect roi)
    {
        List<Rect> bands = collectHorizontalBands(roi);

        if (bands.size() > 0)
        {
            //We find the maximum width
            Rect maxRect = bands.get(0);
            for (int i = 1; i < bands.size(); i++)
            {
                Rect rect = bands.get(i);
                if (rect.height() > maxRect.height())
                    maxRect = rect;
            }
            return maxRect;
        }

        return null;
    }

    List<Rect> collectHorizontalBands(Rect roi)
    {
        Mat edges = new Mat(binaryEdgeMap, roi); //Work on ROI only
        int height = edges.rows();

        //horizontal projection
        Mat verProfile = new Mat(height, 1, CV_32FC1);
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
                    if (grad > 0)
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
            if (top < 60 || height - bottom < 60) continue; //If the band is too close to the top and bottom boundary, we ignore

            Rect rect = new Rect(roi.x(), roi.y()+ top, roi.width(), bottom - top);

            if (!uniformBand(rect)) continue; //If on grayscale image, it is not uniform, we ignore.

            rects.add(rect);
        }

        return rects;
    }

    @Nullable
    Rect verticalCut(Rect roi)
    {
        List<Rect> bands = collectVerticalBands(roi);

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

    List<Rect> collectVerticalBands(Rect roi)
    {
        Mat edges = new Mat(binaryEdgeMap, roi); //Work on ROI only
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
                    if (grad > 0)
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
            if (left < 60 || width - right < 60) continue; //If the band is too close to the left and right boundary, we ignore

            Rect rect = new Rect(roi.x() + left, roi.y(), right - left, roi.height());

            if (!uniformBand(rect)) continue; //If on grayscale image, it is not uniform, we ignore.

            rects.add(rect);
        }

        return rects;
    }

    boolean uniformBand(Rect rect)
    {
        Mat img = new Mat(figure.imageOriginalGray, rect);
        Mat meanMat = new Mat(); Mat stddevMat = new Mat();
        meanStdDev(img, meanMat, stddevMat);

        DoubleRawIndexer meanIndexer = meanMat.createIndexer();
        DoubleRawIndexer stddevIndexer = stddevMat.createIndexer();

        double mean = meanIndexer.get(0);
        double stddev = stddevIndexer.get(0);

        if (stddev > 15.0) return false;

        return true;
    }

    void collectEdgeConnectedComponents()
    {
        //Find and collect all CC's
        edgeCCLabels = new Mat(); edgeCCStats = new Mat();   edgeCCCentroids = new Mat();
        connectedComponentsWithStats(binaryEdgeMap, edgeCCLabels, edgeCCStats, edgeCCCentroids);

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

    void displayPanelCandidates()
    {
        Mat img = figure.imageOriginal.clone();

        //Draw connected components bounding boxes
        for (int i = 0; i < edgeConnectedComponents.size(); i++)
        {
            int left = edgeConnectedComponents.get(i).rectangle.x;
            int top = edgeConnectedComponents.get(i).rectangle.y;
            int width = edgeConnectedComponents.get(i).rectangle.width;
            int height = edgeConnectedComponents.get(i).rectangle.height;

            Point topLeft = new Point(left, top);
            Point bottomRight = new Point(left+width, top+height);
            rectangle(img, topLeft, bottomRight, new Scalar(255, 0, 0, 0) );
        }

        //Draw separatorBands
        for (int i = 0; i < separatorBands.size(); i++)
        {
            Rect rect = separatorBands.get(i);
            rectangle(img, rect, new Scalar(0, 255, 0, 0), CV_FILLED, 8, 0);
        }

        imshow("Panel Candidates", img);
    }

    void detectLineSegments()
    {
        Mat imgEdge = new Mat(structuredEdge.rows(), structuredEdge.cols(), CV_8UC1);
        threshold(structuredEdge, imgEdge, 0.1, 255, THRESH_BINARY_INV);
        imshow("Binary Edges", imgEdge);

        opencv_core.Mat lines = new opencv_core.Mat();		lsd.detect(imgEdge, lines);

        opencv_core.Mat imageLines = new opencv_core.Mat(imgEdge.rows(), imgEdge.cols(), CV_8UC1, opencv_core.Scalar.all(0));
        lsd.drawSegments(imageLines, lines);
        cvtColor(imageLines, imageLines, COLOR_BGRA2GRAY);
        imshow("Gray Lines", imageLines);

        threshold(imageLines, imageLines, 1, 255, THRESH_BINARY_INV);
        imshow("Binary Lines", imageLines);

        //return imageLines;
    }
}
