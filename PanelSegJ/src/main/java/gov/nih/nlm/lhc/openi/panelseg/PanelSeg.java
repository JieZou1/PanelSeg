package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * The class for all panel segmentation algorithms. <p>
 * It defines all constants related to Panel Segmentation, some common functions and initialize and segment functions
 *
 * Created by jzou on 8/25/2016.
 */
public class PanelSeg
{
    public enum Method {
        LabelDetHog, LabelDetHogLeNet5,
        LabelRegHogSvm, LabelRegHogSvmThreshold, LabelRegHogSvmBeam,
        LabelRegHogLeNet5Svm, LabelRegHogLeNet5SvmBeam,

        PanelSplitSantosh, PanelSplitJaylene
    }

    //Below info is collected from LabelStatistics.txt
    static final int labelMinSize = 12;	//The minimum side length of panel labels
    static final int labelMaxSize = 80;	//The maximum side length of panel labels

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

    /**
     * Initialization method, load SVM model, etc.
     * Must call this function before call segment function.
     * @param method
     */
    public static void initialize(Method method) throws Exception
    {
        switch (method)
        {
            case LabelDetHog: return;

            case LabelRegHogSvm:
            case LabelRegHogSvmThreshold:
            case LabelRegHogSvmBeam:
            {
                LabelClassifyHogSvm.initialize("svm_model_rbf_32.0_0.0078125_96.3");
                return;
            }

            case LabelDetHogLeNet5:
            {
                LabelClassifyLeNet5.initialize();
                return;
            }

            case LabelRegHogLeNet5Svm:
            case LabelRegHogLeNet5SvmBeam:
            {
                LabelClassifyLeNet5.initialize();
                LabelClassifyHogSvm.initialize("svm_model_rbf_8.0_0.03125");
                return;
            }

            case PanelSplitSantosh: return;
            case PanelSplitJaylene: return;

            default:
            {
                throw new Exception("Unknown Method!!");
            }
        }
    }

    /**
     * Entry function for all Panel Segmentation methods.
     * The consuming clients just need to know this function.
     * @param image
     * @param method
     * @return
     */
    public static List<Panel> segment(opencv_core.Mat image, Method method) throws Exception
    {
        Figure figure = new Figure(image); //Common initializations for all segmentation method.

        switch (method) {
            case LabelDetHog: {
                LabelDetectHog detectHog = new LabelDetectHog(figure);
                detectHog.hoGDetect();        //HoG Detection, detected patches are stored in hogDetectionResult
                detectHog.mergeDetectedLabelsSimple();  //Merge all hogDetectionResult to panels

                return figure.getSegResultWithoutPadding();
            }

            case LabelRegHogSvm: {
                LabelDetectHog detectHog = new LabelDetectHog(figure);
                detectHog.hoGDetect();        //HoG Detection, detected patches are stored in hogDetectionResult
                detectHog.mergeDetectedLabelsSimple();  //Merge all hogDetectionResult to panels

                //Do label classification with SVM
                LabelClassifyHogSvm classifySvm = new LabelClassifyHogSvm(figure);
                classifySvm.svmClassification();
                classifySvm.mergeRecognitionLabelsSimple();

                return figure.getSegResultWithoutPadding();
            }

            case LabelRegHogSvmThreshold: {
                LabelDetectHog detectHog = new LabelDetectHog(figure);
                detectHog.hoGDetect();        //HoG Detection, detected patches are stored in hogDetectionResult
                detectHog.mergeDetectedLabelsSimple();  //Merge all hogDetectionResult to panels

                //Do label classification with SVM
                LabelClassifyHogSvm classifySvm = new LabelClassifyHogSvm(figure);
                classifySvm.svmClassification();
                classifySvm.mergeRecognitionLabelsSimple();
                classifySvm.threshold(0.98);

                return figure.getSegResultWithoutPadding();
            }

            case LabelRegHogSvmBeam: {
                LabelDetectHog detectHog = new LabelDetectHog(figure);
                detectHog.hoGDetect();        //HoG Detection, detected patches are stored in hogDetectionResult
                detectHog.mergeDetectedLabelsSimple();  //Merge all hogDetectionResult to panels

                //Do label classification with SVM
                LabelClassifyHogSvm classifySvm = new LabelClassifyHogSvm(figure);
                classifySvm.svmClassification();

                LabelBeamSearch beamSearch = new LabelBeamSearch(figure);
                beamSearch.search();

                return figure.getSegResultWithoutPadding();
            }

            case LabelDetHogLeNet5: {
                LabelDetectHog detectHog = new LabelDetectHog(figure);
                detectHog.hoGDetect();        //HoG Detection, detected patches are stored in hogDetectionResult
                detectHog.mergeDetectedLabelsSimple();  //Merge all hogDetectionResult to panels

                //Remove false alarms with LeNet5 model
                LabelClassifyLeNet5 classifyLeNet5 = new LabelClassifyLeNet5(figure);
                classifyLeNet5.LeNet5Classification();    //SVM classification of each detected patch in figure.panels.
                classifyLeNet5.removeFalseAlarms(0.5);

                return figure.getSegResultWithoutPadding();
            }

            case LabelRegHogLeNet5Svm: {
                LabelDetectHog detectHog = new LabelDetectHog(figure);
                detectHog.hoGDetect();        //HoG Detection, detected patches are stored in hogDetectionResult
                detectHog.mergeDetectedLabelsSimple();  //Merge all hogDetectionResult to panels

                LabelClassifyLeNet5 classifyLeNet5 = new LabelClassifyLeNet5(figure);
                classifyLeNet5.LeNet5Classification();    //SVM classification of each detected patch in figure.panels.

                //Do label classification with HoG-SVM
                LabelClassifyHogSvm classifySvm = new LabelClassifyHogSvm(figure);
                classifySvm.svmClassificationWithLeNet5();
                classifySvm.mergeRecognitionLabelsSimple();

                return figure.getSegResultWithoutPadding();
            }

            case LabelRegHogLeNet5SvmBeam: {
                LabelDetectHog detectHog = new LabelDetectHog(figure);
                detectHog.hoGDetect();        //HoG Detection, detected patches are stored in hogDetectionResult
                detectHog.mergeDetectedLabelsSimple();  //Merge all hogDetectionResult to panels

                LabelClassifyLeNet5 classifyLeNet5 = new LabelClassifyLeNet5(figure);
                classifyLeNet5.LeNet5Classification();    //SVM classification of each detected patch in figure.panels.

                //Do label classification with HoG-SVM
                LabelClassifyHogSvm classifySvm = new LabelClassifyHogSvm(figure);
                classifySvm.svmClassificationWithLeNet5();
                //classifySvm.mergeRecognitionLabelsSimple();

                LabelBeamSearch beamSearch = new LabelBeamSearch(figure);
                beamSearch.search();

                return figure.getSegResultWithoutPadding();
            }

            case PanelSplitSantosh: {
                PanelSplitSantosh splitSantosh = new PanelSplitSantosh(figure);
                splitSantosh.split();
                return figure.getSegResultWithoutPadding();
            }

            case PanelSplitJaylene: {
                PanelSplitJaylene splitJaylene = new PanelSplitJaylene(figure);
                splitJaylene.split();
                return figure.getSegResultWithoutPadding();
            }

            default:
            {
                throw new Exception("Unknown Method!!");
            }
        }
    }

    public static List<Panel> segment(String imageFile, Method method) throws Exception
    {
        opencv_core.Mat image = imread(imageFile, CV_LOAD_IMAGE_COLOR);
        return segment(image, method);
    }
    /**
     * The entrance function to perform segmentation.
     * It simply converts the buffered image to Mat, and then calls segment(Mat image, Method method) function.
     *
     * NOTICE: because converting from BufferedImage to Mat requires actual copying of the image data, it is inefficient.
     * It is recommended to avoid using this function if opencv_core.Mat type can be used.
     *
     */
    public static List<Panel> segment(BufferedImage buffered_image, Method method) throws Exception
    {
        opencv_core.Mat image = AlgOpenCVEx.bufferdImg2Mat(buffered_image);
        return segment(image, method);
    }

    /**
     * Sort candidates and remove largely overlapped candidates. <p>
     * Largely here means the overlapping area is over half of the area of the candidates which have higher scores.
     * @param candidates
     * @return
     */
    static ArrayList<Panel> RemoveOverlappedLabelCandidates(ArrayList<Panel> candidates)
    {
        if (candidates == null || candidates.size() == 0 || candidates.size() == 1) return candidates;

        candidates.sort(new LabelScoreDescending());

        //Remove largely overlapped candidates
        ArrayList<Panel> results = new ArrayList<>();        results.add(candidates.get(0));
        for (int j = 1; j < candidates.size(); j++)
        {
            Panel obj = candidates.get(j);            Rectangle obj_rect = obj.labelRect;
            double obj_area = obj_rect.width * obj_rect.height;

            //Check with existing ones, if significantly overlapping with existing ones, ignore
            Boolean overlapping = false;
            for (int k = 0; k < results.size(); k++)
            {
                Rectangle result_rect = results.get(k).labelRect;
                Rectangle intersection = obj_rect.intersection(result_rect);
                if (intersection.isEmpty()) continue;
                double intersection_area = intersection.width * intersection.height;
                double result_area = result_rect.width * result_rect.height;
                if (intersection_area > obj_area / 2 || intersection_area > result_area / 2)
                {
                    overlapping = true; break;
                }
            }
            if (!overlapping) results.add(obj);
        }
        return results;
    }

}
