package gov.nih.nlm.lhc.openi.panelseg;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_imgcodecs;

import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * The core class for holding all information about a Figure.
 * The design is: different algorithms must have a Figure field, which is constructed during the algorithm instance construction;
 * the algorithm takes some fields in Figure object as inputs and then save the results to some other fields of the Figure object.
 *
 * Created by jzou on 8/25/2016.
 */
class Figure
{
    Mat imageColor;		//The original figure image, has to be BGR image
    Mat imageGray;	//The gray image converted from original BGR image
    Mat imageGrayInverted;	//The inverted gray image
    int imageWidth, imageHeight;

    List<Panel> panels;	//The panels of this figure, either loaded from GT data or segmented by an algorithm.

    /**
     * Constructor help function, used by all ctors.
     * @param img BGR image
     */
    private void createFigure(Mat img)
    {
        imageColor = new Mat(img);
        imageGray = new Mat();		cvtColor(imageColor, imageGray, CV_BGR2GRAY);
        imageWidth = imageColor.cols(); imageHeight = imageColor.rows();

        panels = new ArrayList<>();
    }

    /**
     * ctor, from a BGR imageColor
     * imageColor, imageGray, imageWidth and imageHeight are all initialized to the right values.
     * panels is also instantiated as an empty ArrayList.
     * @param img
     */
    Figure(Mat img)
    {
        createFigure(img);
    }

    /**
     * ctor, from a filePath
     * imageColor, imageGray, imageWidth and imageHeight are all initialized to the right values.
     * panels is also instantiated as an empty ArrayList.
     * @param imgPath
     */
    Figure(Path imgPath)
    {
        opencv_core.Mat img = opencv_imgcodecs.imread(imgPath.toString(), opencv_imgcodecs.CV_LOAD_IMAGE_COLOR);
        createFigure(img);
    }

    /**
     * ctor, from a filePath String
     * imageColor, imageGray, imageWidth and imageHeight are all initialized to the right values.
     * panels is also instantiated as an empty ArrayList.
     * @param imgPath
     */
    Figure(String imgPath)
    {
        opencv_core.Mat img = opencv_imgcodecs.imread(imgPath, opencv_imgcodecs.CV_LOAD_IMAGE_COLOR);
        createFigure(img);
    }

    /**
     * ctor, from a filePath
     * imageColor, imageGray, imageWidth and imageHeight are all initialized to the right values.
     * panels is also set.
     * @param imgPath
     */
    Figure(Path imgPath, List<Panel> panels)
    {
        opencv_core.Mat img = opencv_imgcodecs.imread(imgPath.toString(), opencv_imgcodecs.CV_LOAD_IMAGE_COLOR);
        createFigure(img);
        this.panels = panels;
    }

    /**
     * Crop the label patches in original color image.
     * result is saved in labelPatch
     */
    void cropLabelPatches()
    {
        //Pad the original image, such that we could be sure the patch is cropped.
        int padding = 150;
        opencv_core.Mat imgPadded = new opencv_core.Mat();
        opencv_core.copyMakeBorder(imageColor, imgPadded, padding, padding, padding, padding, opencv_core.BORDER_CONSTANT, new opencv_core.Scalar());

        for (Panel panel : panels) {
            if (panel.labelRect == null || panel.labelRect.isEmpty()) continue;

            //Make it square
            int x = panel.labelRect.x, y = panel.labelRect.y, w = panel.labelRect.width, h = panel.labelRect.height;
            if (w > h) {
                int h_new = w;
                int c = y + h / 2;
                y = c - h_new / 2;
                h = h_new;
            } else {
                int w_new = h;
                int c = x + w / 2;
                x = c - w_new / 2;
                w = w_new;
            }

            //Expand 15% in each direction
//            x -= (int) (w * 0.15 + 0.5);
//            w = (int) (w * 1.3 + 0.5);
//            y -= (int) (h * 0.15 + 0.5);
//            h = (int) (h * 1.3 + 0.5);

            x += padding; y += padding;

            panel.labelPatch = imgPadded.apply(new Rect(x, y, w, h));
        }
    }

    /**
     * Crop the label patches in gray image and normalize to norm_w x norm_h
     * result is saved in labelGrayNormPatch
     */
    void cropLabelGrayNormPatches(int norm_w, int norm_h)
    {
        //Pad the original image, such that after expanding the label rects, we still could crop the image.
        int padding = 150;
        opencv_core.Mat imgPadded = new opencv_core.Mat();
        opencv_core.copyMakeBorder(imageGray, imgPadded, padding, padding, padding, padding, opencv_core.BORDER_CONSTANT, new opencv_core.Scalar());

        for (Panel panel : panels) {
            if (panel.labelRect == null || panel.labelRect.isEmpty()) continue;

            //Make it square
            int x = panel.labelRect.x, y = panel.labelRect.y, w = panel.labelRect.width, h = panel.labelRect.height;
            if (w > h) {
                int h_new = w;
                int c = y + h / 2;
                y = c - h_new / 2;
                h = h_new;
            } else {
                int w_new = h;
                int c = x + w / 2;
                x = c - w_new / 2;
                w = w_new;
            }

            //Expand 15% in each direction
//            x -= (int) (w * 0.15 + 0.5);
//            w = (int) (w * 1.3 + 0.5);
//            y -= (int) (h * 0.15 + 0.5);
//            h = (int) (h * 1.3 + 0.5);

            x += padding; y += padding;

            Mat patch = imgPadded.apply(new Rect(x, y, w, h));
            panel.labelGrayNormPatch = new Mat();
            resize(patch, panel.labelGrayNormPatch, new Size(norm_w, norm_h));
        }
    }
}
