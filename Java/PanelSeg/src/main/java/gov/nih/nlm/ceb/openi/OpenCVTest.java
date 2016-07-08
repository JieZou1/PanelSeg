package gov.nih.nlm.ceb.openi;

/**
 * Hello world!
 *
 */
public class OpenCVTest 
{
    public static void main( String[] args )
    {
    	String filename = "D:\\Users\\jie\\Openi\\Panel\\data\\Test\\TinySet\\1465-9921-6-6-4.jpg"; 
    	org.bytedeco.javacpp.opencv_core.Mat image = org.bytedeco.javacpp.opencv_imgcodecs.imread(filename);
    	org.bytedeco.javacpp.opencv_highgui.imshow("image", image);
    	org.bytedeco.javacpp.opencv_highgui.waitKey();
        //System.out.println( "Hello World from OpenCV!" );
    }
}
