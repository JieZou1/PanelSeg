package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_ximgproc;

import static org.bytedeco.javacpp.opencv_core.CV_8UC1;
import static org.bytedeco.javacpp.opencv_core.cvInvert;
import static org.bytedeco.javacpp.opencv_core.invert;
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
        //lsd = createLineSegmentDetector();
    }

    PanelSplitEdgeBox(Figure figure) {this.figure = figure; }

    void split()
    {
        imshow("Image", figure.imageOriginal);
        detectStructuredEdge();
        imshow("StructuredEdges", figure.imageStructuredEdge);
        detectLineSegments();

        waitKey();
    }

    void detectStructuredEdge()
    {
        opencv_core.Mat normImg = new opencv_core.Mat();
        figure.imageOriginal.convertTo(normImg, opencv_core.CV_32F,  1.0 / 255.0, 0.0);

        figure.imageStructuredEdge = new opencv_core.Mat();
        se.detectEdges(normImg, figure.imageStructuredEdge);
    }

    

    void detectLineSegments()
    {
        opencv_core.Mat imgEdge = new opencv_core.Mat(figure.imageStructuredEdge.rows(), figure.imageStructuredEdge.cols(), CV_8UC1);
        threshold(figure.imageStructuredEdge, imgEdge, 0.1, 255, THRESH_BINARY_INV);
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
