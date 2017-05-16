package gov.nih.nlm.lhc.openi.panelseg;

import java.awt.*;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;

import static org.bytedeco.javacpp.opencv_core.subtract;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * The core class for holding all information about a Figure.
 * The design is: different algorithms must have a Figure field.
 * the algorithm takes some fields in Figure object as inputs and then save the results to some other fields of the Figure object.
 *
 * Created by jzou on 8/25/2016.
 */
class Figure
{
    static int padding = 50;

    Mat imageOriginal;  //The original image, has to be BGR image
    Mat imageOriginalGray; //The grayscale image of the original image
    int imageOriginalWidth, imageOriginalHeight; //the original image width and height

    Mat imageColor;		//The figure image after padding,
    Mat imageGray;	//The gray image converted from imageColor
    Mat imageGrayInverted;	//The inverted gray image

    int imageWidth, imageHeight; //the image width and height after padding

    List<Panel> panels;	//The panels of this figure, either loaded from GT data or segmented by an algorithm.
    List<List<LabelBeamSearch.BeamItem>> beams; //The beam search results,

    /**
     * Constructor help function, used by all ctors.
     * @param img BGR image
     */
    private void createFigure(Mat img)
    {
        imageOriginal = img;    //Keep a reference to the original image
        imageOriginalGray = new Mat();		cvtColor(imageOriginal, imageOriginalGray, CV_BGR2GRAY);
        imageOriginalWidth = imageOriginal.cols(); imageOriginalHeight = imageOriginal.rows();

        //We pad padding pixels in each directions of the image.
        imageColor = new opencv_core.Mat();
        opencv_core.copyMakeBorder(img, imageColor, padding, padding, padding, padding, opencv_core.BORDER_CONSTANT, new opencv_core.Scalar());

        imageGray = new Mat();		cvtColor(imageColor, imageGray, CV_BGR2GRAY);
        imageWidth = imageColor.cols(); imageHeight = imageColor.rows();

        imageGrayInverted = subtract(Scalar.all(255), imageGray).asMat();

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
    Figure(Path imgPath) {
        createFigure(opencv_imgcodecs.imread(imgPath.toString(), opencv_imgcodecs.CV_LOAD_IMAGE_COLOR));
    }

    /**
     * ctor, from a filePath String
     * imageColor, imageGray, imageWidth and imageHeight are all initialized to the right values.
     * panels is also instantiated as an empty ArrayList.
     * @param imgPath
     */
    Figure(String imgPath)
    {
        createFigure(opencv_imgcodecs.imread(imgPath, opencv_imgcodecs.CV_LOAD_IMAGE_COLOR));
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
        //The image has already been padded, such that after expanding the label rects, we still could crop the image.
//        int padding = 150;
//        opencv_core.Mat imgPadded = new opencv_core.Mat();
//        opencv_core.copyMakeBorder(imageGray, imgPadded, padding, padding, padding, padding, opencv_core.BORDER_CONSTANT, new opencv_core.Scalar());

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

            panel.labelPatch = imageColor.apply(new Rect(x, y, w, h));
        }
    }

    /**
     * Crop the label patches in gray image and normalize to norm_w x norm_h
     * result is saved in labelGrayNormPatch
     */
    void cropLabelGrayNormPatches(int norm_w, int norm_h)
    {
        //The image has already been padded, such that after expanding the label rects, we still could crop the image.
//        int padding = 150;
//        opencv_core.Mat imgPadded = new opencv_core.Mat();
//        opencv_core.copyMakeBorder(imageGray, imgPadded, padding, padding, padding, padding, opencv_core.BORDER_CONSTANT, new opencv_core.Scalar());

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

            Mat patch = imageGray.apply(new Rect(x, y, w, h));
            panel.labelGrayNormPatch = new Mat();
            resize(patch, panel.labelGrayNormPatch, new Size(norm_w, norm_h));
        }
    }

    /**
     * Get the panel segmentation result
     * Notice the labelRect and panelRect is on the padded image
     * @return The detected panels
     */
    List<Panel> getSegResultWithPadding()	{	return panels;	}

    /**
     * Get the panel segmentation result
     * Notice the labelRect and panelRect is converted back to the original image, i.e., no paddings.
     * @return The detected panels
     */
    List<Panel> getSegResultWithoutPadding()
    {
        List<Panel> panels = new ArrayList<>();
        for (Panel panel: this.panels)
        {
            Panel panelNew = new Panel(panel);
            if (panel.labelRect != null && !panel.labelRect.isEmpty())
            {
                panelNew.labelRect = (Rectangle) panel.labelRect.clone();
                panelNew.labelRect.x -= Figure.padding;
                panelNew.labelRect.y -= Figure.padding;
            }
            if (panel.panelRect != null && !panel.panelRect.isEmpty())
            {
                panelNew.panelRect = (Rectangle) panel.panelRect.clone();
                panelNew.panelRect.x -= Figure.padding;
                panelNew.panelRect.y -= Figure.padding;
            }
//            else
//            {   //panel rect is not set, we use label rect to set one.
//                panelNew.panelRect = (Rectangle) panelNew.labelRect.clone();
//                panelNew.panelRect.x -= panelNew.labelRect.width/2;
//                panelNew.panelRect.y -= panelNew.labelRect.height/2;
//                panelNew.panelRect.width = panelNew.labelRect.width*2;
//                panelNew.panelRect.height = panelNew.labelRect.height*2;
//            }
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
        return drawAnnotation(imageColor, panels);
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

            if (panel.panelLabel != null && panel.panelLabel.length() != 0)
            {
                String label = panel.panelLabel;
                double score = ((int)(panel.labelScore*1000+0.5))/1000.0;
                opencv_core.Rect label_rect = AlgOpenCVEx.Rectangle2Rect(panel.labelRect);
                opencv_core.Point bottom_left = new opencv_core.Point(label_rect.x() + label_rect.width(), label_rect.y() + label_rect.height() + 10);
                opencv_imgproc.putText(imgAnnotated, label + " " + Double.toString(score), bottom_left, opencv_imgproc.CV_FONT_HERSHEY_PLAIN, 1, color, 1, 8, false);
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

            if (panel.panelLabel != null && panel.panelLabel.length() != 0)
            {
                String label = panel.panelLabel;
                double score = ((int)(panel.labelScore*1000+0.5))/1000.0;
                opencv_core.Rect label_rect = AlgOpenCVEx.Rectangle2Rect(panel.labelRect);
                opencv_core.Point bottom_left = new opencv_core.Point(label_rect.x() + label_rect.width(), label_rect.y() + label_rect.height() + 10);
                opencv_imgproc.putText(imgAnnotated, label + " " + Double.toString(score), bottom_left, opencv_imgproc.CV_FONT_HERSHEY_PLAIN, 1, color, 1, 8, false);
            }
        }

        {//Draw Style Data
            opencv_core.Scalar scalar = AlgOpenCVEx.getColor(1);
            opencv_core.Point bottom_left = new opencv_core.Point(0, img.rows() + 30);
            opencv_imgproc.putText(imgAnnotated, style, bottom_left, opencv_imgproc.CV_FONT_HERSHEY_PLAIN, 2, scalar, 3, 8, false);
        }

        return imgAnnotated;
    }
}
