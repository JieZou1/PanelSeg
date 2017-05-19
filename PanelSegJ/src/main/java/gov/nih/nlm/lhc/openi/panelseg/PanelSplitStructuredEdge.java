package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.indexer.DoubleRawIndexer;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_ximgproc;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_core.CV_32FC1;
import static org.bytedeco.javacpp.opencv_core.CV_REDUCE_SUM;
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

    List<Rect> separatorBands;  //The separatorBands (vertical and horizontal uniform bands) detected.
    List<Rect> separatorLines;  //The separatorLines (vertical and horizontal high gradient edges) detected.

    static void initialize() throws Exception
    {
        se = opencv_ximgproc.createStructuredEdgeDetection("/Users/jie/projects/PanelSeg/programs/PanelSegJ/models/model.yml.gz");
        lsd = createLineSegmentDetector();
    }

    PanelSplitStructuredEdge(Figure figure) {this.figure = figure; }

    void detectEdges()
    {
        detectStructuredEdge();
        createBinaryEdgeMap();
    }

    void detectStructuredEdge()
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

    void split(opencv_core.Rect roi_orig)
    {
        //Before we do any analysis, we update its bounding box.
        opencv_core.Mat edges = new opencv_core.Mat(figure.binaryEdgeMap, roi_orig); //Work on ROI only
        opencv_core.Rect roi_inside = AlgOpenCVEx.findBoundingbox(edges);
        opencv_core.Rect roi = new opencv_core.Rect(roi_orig.x()+roi_inside.x(), roi_orig.y() + roi_inside.y(), roi_inside.width(), roi_inside.height());

        opencv_core.Rect verBand = verticalCutByBand(roi);
        opencv_core.Rect horBand = horizontalCutByBand(roi);
        opencv_core.Rect verLine = verticalCutByLine(roi);
        opencv_core.Rect horLine = horizontalCutByLine(roi);

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
                separatorBands.add(verBand);

                //Break into 2 zones vertically
                int left, top, right, bottom;
                left = roi_orig.x(); top = roi_orig.y(); right = verBand.x()+verBand.width()/2; bottom = top + roi_orig.height();
                opencv_core.Rect roi1 = new opencv_core.Rect(left, top, right-left, bottom-top);
                split(roi1);

                left = verBand.x()+verBand.width()/2; top = roi_orig.y(); right = roi_orig.x()+roi_orig.width(); bottom = top + roi_orig.height();
                opencv_core.Rect roi2 = new opencv_core.Rect(left, top, right-left, bottom-top);
                split(roi2);
            }
            else
            {
                //Recover to original roi
                horBand = new opencv_core.Rect(roi_orig.x(), horBand.y(), roi_orig.width(), horBand.height());
                separatorBands.add(horBand);

                //Break into 2 zones horizontally
                int left, top, right, bottom;
                left = roi_orig.x(); top = roi_orig.y(); right = roi_orig.x() + roi.width(); bottom = horBand.y()+horBand.height()/2;
                opencv_core.Rect roi1 = new opencv_core.Rect(left, top, right-left, bottom-top);
                split(roi1);

                left = roi_orig.x(); top = horBand.y()+horBand.height()/2; right = roi_orig.x()+roi_orig.width(); bottom = roi_orig.y() + roi_orig.height();
                opencv_core.Rect roi2 = new opencv_core.Rect(left, top, right-left, bottom-top);
                split(roi2);
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
                separatorLines.add(verLine);

                //Break into 2 zones vertically
                int left, top, right, bottom;
                left = roi_orig.x(); top = roi_orig.y(); right = verLine.x()+verLine.width()/2; bottom = top + roi_orig.height();
                opencv_core.Rect roi1 = new opencv_core.Rect(left, top, right-left, bottom-top);
                split(roi1);

                left = verLine.x()+verLine.width()/2; top = roi_orig.y(); right = roi_orig.x()+roi_orig.width(); bottom = top + roi_orig.height();
                opencv_core.Rect roi2 = new opencv_core.Rect(left, top, right-left, bottom-top);
                split(roi2);
            }
            else
            {
                //Recover to original roi
                horLine = new opencv_core.Rect(roi_orig.x(), horLine.y(), roi_orig.width(), horLine.height());
                separatorLines.add(horLine);

                //Break into 2 zones horizontally
                int left, top, right, bottom;
                left = roi_orig.x(); top = roi_orig.y(); right = roi_orig.x() + roi.width(); bottom = horLine.y()+horLine.height()/2;
                opencv_core.Rect roi1 = new opencv_core.Rect(left, top, right-left, bottom-top);
                split(roi1);

                left = roi_orig.x(); top = horLine.y()+horLine.height()/2; right = roi_orig.x()+roi_orig.width(); bottom = roi_orig.y() + roi_orig.height();
                opencv_core.Rect roi2 = new opencv_core.Rect(left, top, right-left, bottom-top);
                split(roi2);
            }
        }
    }

    @Nullable
    opencv_core.Rect horizontalCutByBand(opencv_core.Rect roi)
    {
        List<opencv_core.Rect> bands = collectHorizontalBands(roi);

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

    List<opencv_core.Rect> collectHorizontalBands(opencv_core.Rect roi)
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
    Rect verticalCutByBand(Rect roi)
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
        Mat edges = new Mat(figure.binaryEdgeMap, roi); //Work on ROI only
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
