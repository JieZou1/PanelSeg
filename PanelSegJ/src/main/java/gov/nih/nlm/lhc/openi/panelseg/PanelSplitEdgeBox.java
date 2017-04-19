package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.indexer.FloatArrayIndexer;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.IntRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_ximgproc;

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

    List<CCInfo> edgeConnectedComponents; //The connected components after some merging
    List<Rect> bands;   //The uniform bands detected.

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

        structuredEdge = new Mat();
        opencv_core.copyMakeBorder(edge, structuredEdge, Figure.padding, Figure.padding, Figure.padding, Figure.padding, opencv_core.BORDER_CONSTANT, new opencv_core.Scalar());
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
        splitByEdgeMap(roi);
    }

    void splitByEdgeMap(Rect roi)
    {
        List<Rect> verticalBands = collectVerticalBands(roi);

        bands = new ArrayList<>();
        bands.addAll(verticalBands);
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

        //Find all 0 runs (no edge bands)
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
            if (right - left < 5) continue; //If the band is too narrow, we ignore
            Rect rect = new Rect(roi.x() + left, roi.y(), right - left, roi.height());

            if (!uniformBand(rect)) continue; //If on grayscale image, it is not uniform, we ignore.

            rects.add(rect);
        }

        return rects;
    }

    boolean uniformBand(Rect rect)
    {
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
        Mat img = figure.imageColor.clone();

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

        //Draw bands
        for (int i = 0; i < bands.size(); i++)
        {
            Rect rect = bands.get(i);
            rectangle(img, rect, new Scalar(0, 255, 0, 0) );
        }

        imshow("Panel Candidates", img);
    }

    void splitByProjection()
    {
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
