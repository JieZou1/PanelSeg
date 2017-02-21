package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_core;

import java.awt.*;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * ReFactored from Santosh's Matlab implementation
 *
 * Created by jzou on 11/18/2016.
 */
public class PanelSplitSantosh
{
    final private double RESIZE_FACTOR = 0.4;					//For resizing the image

    Figure figure;

    private opencv_core.Rect roi;	//The ROI of the image
    protected opencv_core.Mat imagePreprocessed;
    private opencv_core.Mat imageLines;	//The detected lines, white lines drawn on black background
    private opencv_core.Mat imageLinesCleaned;  //Cleaning lines by using CLOSE operator to filtering the detected lines.
    private ArrayList<Integer> horPositions, verPositions; //The vertical (horPositions) and horizontal (verPositions) cut positions (segmentation lines)

    PanelSplitSantosh(Figure figure) {
        this.figure = figure;
    }

    void split()
    {
        imagePreprocessed = preProcess();
        imageLines = detectLineSegments();
        imageLinesCleaned = cleanLineSegments();
        lineProfileAnalysis();

        finalizeSegmentation(); //For mapping back to the original image and find panelRects from horPositions and verPositions
    }

    /**
     * Following https://github.com/arnaudgelas/OpenCVExamples/blob/master/cvMat/Statistics/Median/Median.cpp to calculate Median gray value of an image
     * @param img
     * @return the median of the image gray value.
     */
    private int median(opencv_core.Mat img)
    {
        opencv_core.Mat hist = new opencv_core.Mat();
        IntPointer channels = new IntPointer(new int[] {0});
        Mat mask = new Mat();
        IntPointer histSize = new IntPointer(new int[] {256});
        FloatPointer ranges = new FloatPointer(new float[] {0, 256});
        calcHist(img, 1, channels, mask, hist, 1, histSize, ranges, true, false);

        int med = -1, bin = 0;
        double m=(img.rows()*img.cols())/2;
        FloatBuffer buf = hist.createBuffer();
        for (int i=0; i<256 && med<0;i++)
        {
            bin=bin+cvRound(buf.get());
            if (bin>m && med<0) med=i;
        }
        return med;
    }

    /**
     * Preprocess the image <p>
     * 1. Resize to 0.4 <p>
     * 2. Sharpening the image <p>
     * 3. Crop border via Canny edge detection
     * @return The image after preprocessing (resize, sharpening, canny edge detection and finally cropping
     */
    private opencv_core.Mat preProcess()
    {
        opencv_core.Mat imageResized;	//Resized image
        opencv_core.Mat imageSharpened;	//Resized and sharpened image
        opencv_core.Mat imageCanny;		//Canny edge detection result
        opencv_core.Mat imageCropped;	//Resized and sharpened image after removing boards which do not contain edges. This is the preprocessing result.

        final double GAUSSIAN_RADIUS =  4;//, amount = 2;	//For sharpening the image

        {//Resize the image
            opencv_core.Size reduced_size = new opencv_core.Size((int)(figure.imageGray.cols() * RESIZE_FACTOR + 0.5), (int)(figure.imageGray.rows() * RESIZE_FACTOR + 0.5));
            imageResized = new opencv_core.Mat(reduced_size);
            resize(figure.imageGray, imageResized, reduced_size);
            //imwrite("resized.bmp", imageResized);
        }

        {//Sharpen the image
            //Unsharping masking: Use a Gaussian smoothing filter and subtract the smoothed version from the original image (in a weighted way so the values of a constant area remain constant).
            //NOTE: The original Matlab statement is sharp_img = imsharpen(img_in, 'Radius', 4, 'Amount', 2);
            //      OpenCV does not provide this sharpening function, we have to simulate it.
            opencv_core.Mat imageBlurred = new opencv_core.Mat(imageResized.size());
            GaussianBlur(imageResized, imageBlurred, new opencv_core.Size(0, 0), GAUSSIAN_RADIUS);

            imageSharpened = new opencv_core.Mat(imageResized.size());
            addWeighted(imageResized, 1.5, imageBlurred, -0.5, 0, imageSharpened);
            //imwrite("sharpened.bmp", imageSharpened);
        }

        //Convert to gray scale image
        //Since we load image as gray scale, we don't need this step
//		cvtColor(imageSharpened, imageGray, CV_BGR2GRAY);
//		imwrite("gray.bmp", imageGray);

        {//Canny edge detection
            //NOTE: The original Matlab statement is BW = edge(gray_img,'canny')
            //But, OpenCV requires to specify 2 thresholds (high and low).
            //I am following this blog post: http://www.kerrywong.com/2009/05/07/canny-edge-detection-auto-thresholding/ and used 0.66*Median and 1.33*median
            imageCanny = new opencv_core.Mat(imageResized.size());
            int median = median(imageSharpened);
            Canny(imageSharpened, imageCanny, 0.66 * median, 1.33 * median);
            //imwrite("canny.bmp", imageCanny);
        }

        {//Border Crop
            opencv_core.Mat points = new opencv_core.Mat();
            findNonZero(imageCanny, points);
            roi = boundingRect(points);
            //Rectangle rect = new Rectangle(roi.x(), roi.y(), roi.width(), roi.height());
            imageCropped = new opencv_core.Mat(imageSharpened, roi);
            //imwrite("cropped.bmp", imageCropped);
        }

        return imageCropped;
    }

    /**
     * Detect line segments with LSD method
     * @return the detected line segments drawn on a binary image. (black backgraound, white lines)
     */
    private opencv_core.Mat detectLineSegments()
    {
        LineSegmentDetector lsd = createLineSegmentDetector();
        opencv_core.Mat lines = new opencv_core.Mat();		lsd.detect(imagePreprocessed, lines);

        Mat imageLines = new Mat(imagePreprocessed.rows(), imagePreprocessed.cols(), CV_8UC1, Scalar.all(0));
        //imwrite("lines.bmp", imageLines);
        lsd.drawSegments(imageLines, lines);
        cvtColor(imageLines, imageLines, COLOR_BGRA2GRAY);
        threshold(imageLines, imageLines, 1, 255, THRESH_BINARY_INV);
        //imwrite("lines.bmp", imageLines);

        return imageLines;
    }

    /**
     * Use CLOSE morphology to keep only horizontal and vertical line segments
     * @return The filtered lines drawn in a white-on-black gray image.
     */
    private Mat cleanLineSegments()
    {
        Mat imageBoarderdAdded;		//Add boarders back, white foreground, black background
        Mat closedVec = new Mat(), closedHor = new Mat();	//After CLOSE filtering,
        Mat imageLinesCleaned = new Mat();		//lines after cleaning.

        {//Keep boarders
            Mat points = new Mat();
            findNonZero(imageLines, points);
            Rect roi = boundingRect(points);
            imageBoarderdAdded = new Mat(imageLines, roi);
            rectangle(imageBoarderdAdded, new opencv_core.Point(0, 0), new opencv_core.Point(imageBoarderdAdded.cols()-1, imageBoarderdAdded.rows()-1), Scalar.all(0), 2, 8, 0);
            //imwrite("withboarders.bmp", imageBoarderdAdded);
        }

        {//Filtering the image with close operation, to get horizontal and vertical line segments.
            Mat meHor = getStructuringElement(MORPH_RECT, new Size(10, 1));
            Mat meVec = getStructuringElement(MORPH_RECT, new Size(1, 10));
            morphologyEx(imageBoarderdAdded, closedHor, MORPH_CLOSE, meHor);
            morphologyEx(imageBoarderdAdded, closedVec, MORPH_CLOSE, meVec);

            //invert to white on black images
            Mat white = new Mat(imageBoarderdAdded.rows(),imageBoarderdAdded.cols(), imageBoarderdAdded.type(), Scalar.all(255));
            subtract(white, closedHor, closedHor);
            subtract(white, closedVec, closedVec);

            //imwrite("hor.bmp", closedHor);		imwrite("ver.bmp", closedVec);
        }

        addWeighted(closedHor, 0.5, closedVec, 0.5, 0.0, imageLinesCleaned);
        threshold(imageLinesCleaned, imageLinesCleaned, 1, 255, THRESH_BINARY);
        return imageLinesCleaned;
    }

    /**
     * Normalize Vertical/Horizontal Projection profile
     * @param profile input profile
     * @return the normalized profile
     */
    private Mat normalizeProfile(Mat profile)
    {
        Mat profileNormalized = new Mat();		multiply(profile, profile, profileNormalized);

        Mat meanMat = new Mat(), stdMat = new Mat();		meanStdDev(profileNormalized, meanMat, stdMat);
        double mean = ((DoubleBuffer)(meanMat.createBuffer())).get();
        double stddev = ((DoubleBuffer)(stdMat.createBuffer())).get();

        profileNormalized =  divide(subtract(profileNormalized, new Scalar(mean)), stddev).asMat();
        threshold(profileNormalized, profileNormalized, 0, 255, CV_THRESH_TOZERO);

//		{
//		FileStorage file = new FileStorage("profile.txt", FileStorage.WRITE);
//		write(file, "profile", profileNormalized);
//		file.release();
//		}

        return profileNormalized;
    }

    /**
     * Find peaks and their positions from a profile projection.
     * @param profile The input profile. It can be 1XN or NX1 float (CV_32FC1) matrix
     * @param threshold	The minimum peak height
     * @param distance	The minimum distance between adjacent peaks
     * @param maxValues Output, the peak values
     * @param maxIndexes Output, the peak positions.
     */
    private void findPeaks(Mat profile, double threshold, int distance, ArrayList<Double> maxValues, ArrayList<Integer> maxIndexes)
    {
        maxValues.clear(); maxIndexes.clear();

        FloatBuffer profileBuf = profile.createBuffer();
        int limit = profileBuf.limit();
        float[] profileArr = new float[limit]; profileBuf.get(profileArr);

        while (true)
        {
            double max_value = 0; int max_index = 0;
            for (int i = 0; i < limit; i++)
            {
                if (profileArr[i] > max_value)
                {
                    max_value = profileArr[i];
                    max_index = i;
                }
            }

            if (max_value > threshold)
            {
                maxValues.add(max_value);
                maxIndexes.add(max_index);

                for (int i = -distance / 2; i <= distance / 2; i++)
                {
                    int index = max_index + i;
                    if (index < 0 || index >= limit) continue;
                    profileArr[index] = 0;
                }
            }
            else break;
        }
    }

    /**
     * Finalize the candidate boundary lines by filtering the vertical and horizontal line profiles.
     * horPositions Output: the horizontal positions of the vertical candidate cuts (segment lines)
     * verPositions Output: the vertical positions of the horizontal candidate cuts (segment lines)
     */
    private void lineProfileAnalysis()
    {
        {//Analyze horizontal profile
            Mat horProfile = new Mat(1, imageLinesCleaned.cols(), CV_32FC1);
            reduce(imageLinesCleaned, horProfile, 0, CV_REDUCE_SUM, CV_32FC1); //Vertical projection generate Horizontal profile
            horProfile = divide(horProfile, 255).asMat();
            Mat horProfileNorm = normalizeProfile(horProfile);

            double[] max_value = new double[1]; int[] max_index = new int[1];
            DoublePointer max_value_p = new DoublePointer(max_value); IntPointer max_index_p = new IntPointer(max_index);
            minMaxIdx(horProfileNorm, null, max_value_p, null, max_index_p, noArray());
            max_value_p.get(max_value); max_index_p.get(max_index);
            double thresh = (max_value[0]/2)* 0.4;
            int distance = horProfileNorm.cols() < 50 ? horProfileNorm.cols()-10 : 50;

            ArrayList<Double> maxValues = new ArrayList<Double>(); ArrayList<Integer> maxIndexes = new ArrayList<Integer>();
            findPeaks(horProfileNorm, thresh, distance, maxValues, maxIndexes);
            horPositions = maxIndexes;
        }

        {//Analyze vertical profile
            Mat verProfile = new Mat(imageLinesCleaned.rows(), 1, CV_32FC1);
            reduce(imageLinesCleaned, verProfile, 1, CV_REDUCE_SUM, CV_32FC1); //Horizontal projection generate vertical profile
            verProfile = divide(verProfile, 255).asMat();
            Mat verProfileNorm = normalizeProfile(verProfile);

            double[] max_value = new double[1]; int[] max_index = new int[1];
            DoublePointer max_value_p = new DoublePointer(max_value); IntPointer max_index_p = new IntPointer(max_index);
            minMaxIdx(verProfileNorm, null, max_value_p, null, max_index_p, noArray());
            max_value_p.get(max_value); max_index_p.get(max_index);
            double thresh = (max_value[0]/2)* 0.4;
            int distance = verProfileNorm.rows() < 50 ? verProfileNorm.rows() - 10 : 50;

            ArrayList<Double> maxValues = new ArrayList<Double>(); ArrayList<Integer> maxIndexes = new ArrayList<Integer>();
            findPeaks(verProfileNorm, thresh, distance, maxValues, maxIndexes);
            verPositions = maxIndexes;
        }
    }

    /**
     * Find panelRects from horPositions and verPositions and then Map back to the original image
     */
    private void finalizeSegmentation()
    {
        int xOffset = roi.x(), yOffset = roi.y(); //For mapping back to the original image

        //Sort horPositions and verPositions
        Collections.sort(horPositions);
        Collections.sort(verPositions);

        //Find rectangle panels from horPositions and verPositions
        figure.panels = new ArrayList<Panel>();
        //figure.result.panelRects = new ArrayList<Rectangle>();
        for (int i = 0; i < horPositions.size() - 1; i++)
        {
            int hor1 = (int)((horPositions.get(i) + xOffset) / RESIZE_FACTOR + 0.5);
            int hor2 = (int)((horPositions.get(i + 1) + xOffset) / RESIZE_FACTOR + 0.5);
            for (int j = 0; j < verPositions.size() - 1; j++)
            {
                int ver1 = (int)((verPositions.get(j) + yOffset) / RESIZE_FACTOR + 0.5);
                int ver2 = (int)((verPositions.get(j + 1) + yOffset) / RESIZE_FACTOR + 0.5);

                Rectangle rect = new Rectangle(hor1, ver1, hor2-hor1, ver2 - ver1);
                Panel panel = new Panel();
                panel.panelRect = rect;
                //figure.result.panelRects.add(rect);
                figure.panels.add(panel);
            }
        }
    }

}
