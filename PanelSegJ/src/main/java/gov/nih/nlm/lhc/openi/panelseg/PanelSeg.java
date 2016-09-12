package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.CV_FONT_HERSHEY_PLAIN;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

/**
 * The base class for all panel segmentation algorithms. <p>
 * This class is not intended to be instantiated, so we make it abstract.
 *
 * Created by jzou on 8/25/2016.
 */
public abstract class PanelSeg
{
    public enum SegMethed {LabelRegHog}

    /**
     * Entry function for all Panel Segmentation methods.
     * The consuming clients just need to know this function.
     * @param image
     * @param method
     * @return
     */
    public static java.util.List<Panel> segment(opencv_core.Mat image, SegMethed method)
    {
        PanelSeg seg = null;
        switch (method)
        {
            case LabelRegHog: seg = new PanelSegLabelRegHog(); break;
        }
        seg.segment(image);
        return seg.getSegmentationResult();
    }

    //All possible panel label chars, 'c', 'k', 'o', 'p', 's', 'u', 'v' 'w', 'x', 'y', 'z' no difference between upper and lower cases.
    static final char[] labelChars = {
            'a', 'A', 'b', 'B', 'c', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H',
            'i', 'I', 'j', 'J', 'k', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'p', 'q', 'Q',
            'r', 'R', 's', 't', 'T', 'u', 'v', 'w', 'x', 'y', 'z',
            '1', '2', '3', '4', '5', '6', '7', '8', '9'
    };

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
     * Call getSegmentationResult* functions to retrieve result in different format.
     */
    void segment(String image_file_path)
    {
        opencv_core.Mat image = imread(image_file_path, CV_LOAD_IMAGE_COLOR);
        segment(image);
    }

    /**
     * The entrance function to perform segmentation.
     * Call getSegmentationResult* functions to retrieve result in different format.
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
     * @return The detected panels
     */
    public java.util.List<Panel> getSegmentationResult()	{	return figure.panels;	}

    /**
     * Get the panel segmentation result by drawing the panel boundaries on the image
     * @return the image with panel boundaries superimposed on it.
     */
    public opencv_core.Mat getSegmentationResultInMat()
    {
        opencv_core.Mat img = figure.imageColor.clone();
        for (Panel panel : figure.panels)
        {
            if (panel.panelRect != null )
            {
                Rectangle rect = panel.panelRect;
                rectangle(img,
                        new opencv_core.Point(rect.x, rect.y), new opencv_core.Point(rect.x + rect.width, rect.y + rect.height),
                        opencv_core.Scalar.RED, 3, 8, 0);
            }
            if (panel.labelRect != null)
            {
                Rectangle rect = panel.labelRect;
                rectangle(img,
                        new opencv_core.Point(rect.x, rect.y), new opencv_core.Point(rect.x + rect.width, rect.y + rect.height),
                        opencv_core.Scalar.BLUE, 1, 8, 0);

                if (panel.panelLabel != "")
                {
                    putText(img, panel.panelLabel,
                            new opencv_core.Point(panel.labelRect.x + panel.labelRect.width, panel.labelRect.y + panel.labelRect.height),
                            CV_FONT_HERSHEY_PLAIN, 2.0, opencv_core.Scalar.BLUE, 3, 8, false);
                }
            }
        }

        return img;
    }


}
