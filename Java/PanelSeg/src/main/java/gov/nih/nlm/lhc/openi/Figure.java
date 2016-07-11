package gov.nih.nlm.lhc.openi;

import java.util.ArrayList;
import org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * The core class for holding all information about a Figure. 
 * The design is: different algorithms must have a Figure field, which is constructed during the algorithm instance construction; 
 * the algorithm takes some fields in Figure object as inputs and then save the results to some other fields of the Figure object.  
 * 
 * @author Jie Zou
 *
 */
public class Figure 
{
	Mat image;		//The original figure image, has to be BGR image
	Mat imageGray;	//The gray image converted from original BGR image
	//Mat imageGrayInverted;	//The inverted gray image
	int imageWidth, imageHeight;
	
	ArrayList<Panel> panels;	//The panels of this figure, either loaded from GT data or segmented by an algorithm.
	
	/**
	 * ctor, from a BGR image
	 * image, imageGray, imageWidth and imageHeight are all initialized to the right values.
	 * panels is also instantiated as an empty ArrayList.
	 * @param img
	 */
	Figure(Mat img)
	{
		image = new Mat(img);
		imageGray = new Mat();		cvtColor(image, imageGray, CV_BGR2GRAY);
		imageWidth = image.cols(); imageHeight = image.rows();
		
		panels = new ArrayList<Panel>();		
	}

}
