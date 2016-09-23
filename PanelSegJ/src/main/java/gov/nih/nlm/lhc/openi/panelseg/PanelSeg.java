package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * The base class for all panel segmentation algorithms. <p>
 * This class is not intended to be instantiated, so we make it abstract.
 *
 * Created by jzou on 8/25/2016.
 */
public abstract class PanelSeg
{
    public enum SegMethod {LabelRegHog, LabelRegHogSvm, LabelRegHogSvmThresholding, LabelRegHogSvmBeam}

    /**
     * Initialization method, load SVM model, etc.
     * Must call this function before call segment function.
     * @param method
     */
    public static void initialize(SegMethod method)
    {
        switch (method)
        {
            case LabelRegHog: break;
            case LabelRegHogSvm: PanelSegLabelRegHogSvm.initialize(); break;
            case LabelRegHogSvmThresholding: PanelSegLabelRegHogSvmThresholding.initialize(); break;
            case LabelRegHogSvmBeam: PanelSegLabelRegHogSvmBeam.initialize(); break;
        }
    }

    /**
     * Entry function for all Panel Segmentation methods.
     * The consuming clients just need to know this function.
     * @param image
     * @param method
     * @return
     */
    public static List<Panel> segment(opencv_core.Mat image, SegMethod method)
    {
        PanelSeg seg = null;
        switch (method)
        {
            case LabelRegHog: seg = new PanelSegLabelRegHog(); break;
            case LabelRegHogSvm: seg = new PanelSegLabelRegHogSvm();  break;
            case LabelRegHogSvmThresholding: seg = new PanelSegLabelRegHogSvmThresholding();  break;
            case LabelRegHogSvmBeam: seg = new PanelSegLabelRegHogSvmBeam();  break;
        }
        seg.segment(image);
        return seg.getSegResultWithoutPadding();
    }

    public static List<Panel> segment(String imageFile, SegMethod method)
    {
        opencv_core.Mat image = imread(imageFile, CV_LOAD_IMAGE_COLOR);
        return segment(image, method);
    }

    //All possible panel label chars
    static final char[] labelChars = {
            'a', 'A', 'b', 'B', 'c', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H',
            'i', 'I', 'j', 'J', 'k', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'p', 'q', 'Q',
            'r', 'R', 's', 't', 'T', 'u', 'v', 'w', 'x', 'y', 'z',
            '1', '2', '3', '4', '5', '6', '7', '8', '9'
    };

    /**
     * 'c', 'k', 'o', 'p', 's', 'u', 'v' 'w', 'x', 'y', 'z' no difference between upper and lower cases.
     * @param c
     * @return
     */
    static boolean isCaseSame(char c)
    {
        return (c == 'c' ||
                c == 'k' ||
                c == 'o' ||
                c == 'p' ||
                c == 's' ||
                c == 'u' ||
                c == 'v' ||
                c == 'w' ||
                c == 'x' ||
                c == 'y' ||
                c == 'z');
    }

    /**
     * Convert label char to folder name. 'a' and 'A' are 2 different char, but a and A folders are the same.
     * @param labelChar
     * @return
     */
    static String getLabelCharFolderName(char labelChar) {
        //Special treatment for those identical upper and lower chars
        Character labelCharLower = Character.toLowerCase(labelChar);
        if (labelCharLower == 'c') return "c";
        if (labelCharLower == 'k') return "k";
        if (labelCharLower == 'o') return "o";
        if (labelCharLower == 'p') return "p";
        if (labelCharLower == 's') return "s";
        if (labelCharLower == 'u') return "u";
        if (labelCharLower == 'v') return "v";
        if (labelCharLower == 'w') return "w";
        if (labelCharLower == 'x') return "x";
        if (labelCharLower == 'y') return "y";
        if (labelCharLower == 'z') return "z";

        return Character.isUpperCase(labelChar) ? labelChar + "_" : Character.toString(labelChar);
    }

    //Below info is collected from LabelStatistics.txt
    static final int labelMinSize = 12;	//The minimum side length of panel labels
    static final int labelMaxSize = 80;	//The maximum side length of panel labels

    protected Figure figure;

    abstract void segment(opencv_core.Mat image);

    /**
     * The entrance function to perform panel segmentation. <p>
     * It simply loads the image from the file, and then calls segment(Mat image) function.
     * Call getSegResultWithPadding* functions to retrieve result in different format.
     */
    void segment(String image_file_path)
    {
        opencv_core.Mat image = imread(image_file_path, CV_LOAD_IMAGE_COLOR);
        segment(image);
    }

    /**
     * The entrance function to perform segmentation.
     * Call getSegResultWithPadding* functions to retrieve result in different format.
     * It simply converts the buffered image to Mat, and then calls segment(Mat image) function.
     *
     * NOTICE: because converting from BufferedImage to Mat requires actual copying of the image data, it is inefficient.
     * It is recommended to avoid using this function if opencv_core.Mat type can be used.
     *
     */
    void segment(BufferedImage buffered_image) throws Exception
    {
        opencv_core.Mat image = AlgOpenCVEx.bufferdImg2Mat(buffered_image);
        segment(image);
    }

    /**
     * Get the panel segmentation result
     * Notice the labelRect and panelRect is on the padded image
     * @return The detected panels
     */
    List<Panel> getSegResultWithPadding()	{	return figure.panels;	}

    /**
     * Get the panel segmentation result
     * Notice the labelRect and panelRect is converted back to the original image, i.e., no paddings.
     * @return The detected panels
     */
    List<Panel> getSegResultWithoutPadding()
    {
        List<Panel> panels = new ArrayList<>();
        for (Panel panel: figure.panels)
        {
            Panel panelNew = new Panel(panel);
            if (panelNew.labelRect != null && !panelNew.labelRect.isEmpty())
            {
                panelNew.labelRect = (Rectangle) panel.labelRect.clone();
                panelNew.labelRect.x -= Figure.padding;
                panelNew.labelRect.y -= Figure.padding;
            }
            if (panelNew.panelRect != null && !panelNew.panelRect.isEmpty())
            {
                panelNew.panelRect = (Rectangle) panel.panelRect.clone();
                panelNew.panelRect.x -= Figure.padding;
                panelNew.panelRect.y -= Figure.padding;
            }
            panels.add(panelNew);
        }
        return panels;
    }

    /**
     * Get the panel segmentation result by drawing the panel boundaries on the image
     * @return the image with panel boundaries superimposed on it.
     */
    opencv_core.Mat getSegResultWithPaddingInMat()
    {
        return drawAnnotation(figure.imageColor, figure.panels);
//        opencv_core.Mat img = figure.imageColor.clone();
//        for (Panel panel : figure.panels)
//        {
//            if (panel.panelRect != null )
//            {
//                Rectangle rect = panel.panelRect;
//                rectangle(img,
//                        new opencv_core.Point(rect.x, rect.y), new opencv_core.Point(rect.x + rect.width, rect.y + rect.height),
//                        opencv_core.Scalar.RED, 3, 8, 0);
//            }
//            if (panel.labelRect != null)
//            {
//                Rectangle rect = panel.labelRect;
//                rectangle(img,
//                        new opencv_core.Point(rect.x, rect.y), new opencv_core.Point(rect.x + rect.width, rect.y + rect.height),
//                        opencv_core.Scalar.BLUE, 1, 8, 0);
//
//                if (panel.panelLabel != "")
//                {
//                    putText(img, panel.panelLabel,
//                            new opencv_core.Point(panel.labelRect.x + panel.labelRect.width, panel.labelRect.y + panel.labelRect.height),
//                            CV_FONT_HERSHEY_PLAIN, 2.0, opencv_core.Scalar.BLUE, 3, 8, false);
//                }
//            }
//        }
//
//        return img;
    }

    /**
     * Draw annotation onto the img for previewing and saving purpose
     * @param img
     * @param panels
     * @return
     */
    static opencv_core.Mat drawAnnotation(opencv_core.Mat img, List<Panel> panels)
    {
        opencv_core.Mat imgAnnotated = new opencv_core.Mat();
        opencv_core.copyMakeBorder(img, imgAnnotated, 0, 50, 0, 50, opencv_core.BORDER_CONSTANT, new opencv_core.Scalar());

        //Draw bounding box first
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            opencv_core.Scalar color = AlgOpenCVEx.getColor(i);

            if (panel.panelRect != null && !panel.panelRect.isEmpty())
            {
                opencv_core.Rect panel_rect = AlgOpenCVEx.Rectangle2Rect(panel.panelRect);
                opencv_imgproc.rectangle(imgAnnotated, panel_rect, color, 3, 8, 0);
            }

            if (panel.labelRect != null && !panel.labelRect.isEmpty())
            {
                opencv_core.Rect label_rect = AlgOpenCVEx.Rectangle2Rect(panel.labelRect);
                opencv_imgproc.rectangle(imgAnnotated, label_rect, color, 1, 8, 0);
            }
        }

        //Draw labels to make the text stand out.
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            opencv_core.Scalar color = AlgOpenCVEx.getColor(i);

            if (panel.panelLabel.length() != 0)
            {
                opencv_core.Rect label_rect = AlgOpenCVEx.Rectangle2Rect(panel.labelRect);
                opencv_core.Point bottom_left = new opencv_core.Point(label_rect.x() + label_rect.width(), label_rect.y() + label_rect.height() + 10);
                opencv_imgproc.putText(imgAnnotated, panel.panelLabel + Double.toString(panel.labelScore), bottom_left, opencv_imgproc.CV_FONT_HERSHEY_PLAIN, 1, color, 1, 8, false);
            }
        }

        return imgAnnotated;
    }

    /**
     * Draw annotation onto the img for previewing and saving purpose
     * @param img
     * @param panels
     * @return
     */
    static opencv_core.Mat drawAnnotation(opencv_core.Mat img, List<Panel> panels, String style)
    {
        opencv_core.Mat imgAnnotated = new opencv_core.Mat();
        opencv_core.copyMakeBorder(img, imgAnnotated, 0, 100, 0, 50, opencv_core.BORDER_CONSTANT, new opencv_core.Scalar());

        //Draw bounding box first
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            opencv_core.Scalar color = AlgOpenCVEx.getColor(i);

            opencv_core.Rect panel_rect = AlgOpenCVEx.Rectangle2Rect(panel.panelRect);
            opencv_imgproc.rectangle(imgAnnotated, panel_rect, color, 3, 8, 0);

            if (panel.panelLabel.length() != 0)
            {
                opencv_core.Rect label_rect = AlgOpenCVEx.Rectangle2Rect(panel.labelRect);
                opencv_imgproc.rectangle(imgAnnotated, label_rect, color, 1, 8, 0);
            }
        }

        //Draw labels to make the text stand out.
        for (int i = 0; i < panels.size(); i++)
        {
            Panel panel = panels.get(i);
            opencv_core.Scalar color = AlgOpenCVEx.getColor(i);

            if (panel.panelLabel.length() != 0)
            {
                opencv_core.Rect label_rect = AlgOpenCVEx.Rectangle2Rect(panel.labelRect);
                opencv_core.Point bottom_left = new opencv_core.Point(label_rect.x() + label_rect.width(), label_rect.y() + label_rect.height() + 50);
                opencv_imgproc.putText(imgAnnotated, panel.panelLabel, bottom_left, opencv_imgproc.CV_FONT_HERSHEY_PLAIN, 5, color, 3, 8, false);
            }
        }

        {//Draw Style Data
            opencv_core.Scalar scalar = AlgOpenCVEx.getColor(1);
            opencv_core.Point bottom_left = new opencv_core.Point(0, img.rows() + 100);
            opencv_imgproc.putText(imgAnnotated, style, bottom_left, opencv_imgproc.CV_FONT_HERSHEY_PLAIN, 2, scalar, 3, 8, false);
        }

        return imgAnnotated;
    }

}
