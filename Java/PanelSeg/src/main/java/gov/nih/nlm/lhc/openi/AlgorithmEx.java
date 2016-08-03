package gov.nih.nlm.lhc.openi;

import java.util.ArrayList;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.awt.*;
import org.w3c.dom.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class AlgorithmEx 
{
	/**
	 * Some random testing codes
	 * @param args
	 */
    public static void main( String[] args )
    {
    	String filename;
    	
    	if (args.length == 0)
        	filename = "D:\\Users\\jie\\Openi\\Panel\\data\\Test\\TinySet\\1465-9921-6-6-4.jpg";
    	else if (args.length == 1)
        	filename = args[0];
    	else return;
    	
    	org.bytedeco.javacpp.opencv_core.Mat image = org.bytedeco.javacpp.opencv_imgcodecs.imread(filename);
    	org.bytedeco.javacpp.opencv_highgui.imshow("image", image);
    	org.bytedeco.javacpp.opencv_highgui.waitKey();
        System.out.println( "Hello World from OpenCV!" );
    }

	/**
	 * Very inefficient function to convert BufferImage between types
	 * @param src
	 * @param bufImgType
	 * @return
	 */
	private static BufferedImage convert(BufferedImage src, int bufImgType) 
	{
	    BufferedImage img= new BufferedImage(src.getWidth(), src.getHeight(), bufImgType);
	    Graphics2D g2d= img.createGraphics();
	    g2d.drawImage(src, 0, 0, null);
	    g2d.dispose();
	    return img;
	}
	
	/**
	 * To convert BufferedImage to Mat format. Currently, used a very inefficient method. 
	 * @param in
	 * @return Mat image in BGR format
	 */
	static Mat bufferdImg2Mat(BufferedImage in)
	{
		if (in.getType() != BufferedImage.TYPE_INT_RGB)		in = convert(in, BufferedImage.TYPE_INT_RGB);
		
		Mat out;
		byte[] data;         int r, g, b;          int height = in.getHeight(), width = in.getWidth();
		if(in.getType() == BufferedImage.TYPE_INT_RGB || in.getType() == BufferedImage.TYPE_INT_ARGB)
        {
			out = new Mat(height, width, CV_8UC3);
			data = new byte[height * width * (int)out.elemSize()];
			int[] dataBuff = in.getRGB(0, 0, width, height, null, 0, width);
			for(int i = 0; i < dataBuff.length; i++)
			{
				data[i*3 + 2] = (byte) ((dataBuff[i] >> 16) & 0xFF);
				data[i*3 + 1] = (byte) ((dataBuff[i] >> 8) & 0xFF);
				data[i*3] = (byte) ((dataBuff[i] >> 0) & 0xFF);
			}
			out.data().put(data);
        }
		else
		{
			out = new Mat(height, width,  CV_8UC1);
			data = new byte[height * width * (int)out.elemSize()];
			int[] dataBuff = in.getRGB(0, 0, width, height, null, 0, width);
			for(int i = 0; i < dataBuff.length; i++)
			{
				r = (byte) ((dataBuff[i] >> 16) & 0xFF);
                g = (byte) ((dataBuff[i] >> 8) & 0xFF);
                b = (byte) ((dataBuff[i] >> 0) & 0xFF);
                data[i] = (byte)((0.21 * r) + (0.71 * g) + (0.07 * b)); //luminosity
			}
			out.data().put(data);
		}
		return out;
    }

	/**
	 * Convert Mat image to BufferedImage <p>
	 * BufferedImage is either in TYPE_BYTE_GRAY or TYPE_INT_RGB format.
	 * @param in
	 * @return 
	 */
	static BufferedImage mat2BufferdImg(Mat in)
    {
		int width = in.cols(), height = in.rows();
        BufferedImage out;
        byte[] data = new byte[width * height * (int)in.elemSize()];
        int type;
        in.data().get(data);

        if(in.channels() == 1)
            type = BufferedImage.TYPE_BYTE_GRAY;
        else
            type = BufferedImage.TYPE_3BYTE_BGR;

        out = new BufferedImage(width, height, type);

        out.getRaster().setDataElements(0, 0, width, height, data);
        out = convert(out, BufferedImage.TYPE_INT_RGB);
//        try {
//			ImageIO.write(out, "jpg", new File("temp.jpg"));
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
        return out;
    } 

	/**
	 * return a child node of the parent node based on its tag
	 * @param parent
	 * @param tag
	 * @return
	 */
	static Node getChildNode(Node parent, String tag)
	{
		String nodeName;
		NodeList children = parent.getChildNodes();
		for (int j = 0; j < children.getLength(); j++)
		{
			Node child = children.item(j);
			nodeName = child.getNodeName();
			if (tag != nodeName) continue;
			
			return child;
		}
		return null;
	}

	/**
	 * Find the bounding box of a binary image. White as foreground
	 * @param bina
	 * @return
	 */
	static Rect findBoundingbox(Mat bina)
	{
		Mat points = new Mat();	findNonZero(bina,points);
		Rect minRect=boundingRect(points);	
		return minRect;
	}
	
	/**
	 * Crop the ROI from the image
	 * @param image
	 * @param roi
	 * @return
	 */
	static Mat cropImage(Mat image, Rectangle roi)
	{
		Rect rect = new Rect(roi.x, roi.y, roi.width, roi.height);
		return image.apply(rect);
	}

	static int findMaxIndex(double[] array)
	{
		int maxIndex= 0; double maxValue = array[0];
		for (int i = 1; i < array.length; i++)
		{
			if (array[i] > maxValue)
			{
				maxIndex = i; maxValue = array[i];
			}
		}
		return maxIndex;
	}

	/**
	 * Convert java.awt.Rectangle to opencv_core.Rect
	 * @param rectangle
	 * @return
	 */
	static Rect Rectangle2Rect(Rectangle rectangle)
	{
		Rect rect = new Rect(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
		return rect;
	}
	
	/**
	 * Get a color, for drawing on Mat (image)
	 * @param i
	 * @return
	 */
	static Scalar getColor(int i)
	{
		Color colors[] = new Color[]
				{
						Color.BLUE,
						Color.CYAN,
						Color.GREEN,
						Color.MAGENTA,
						Color.ORANGE,
						Color.PINK,
						Color.RED,
						Color.YELLOW,
				};
		Color color = colors[i%colors.length];
		Scalar scalar = new Scalar(color.getBlue(), color.getGreen(), color.getRed(), 0);
		return scalar;
	}
	
	/**
	 * Collect all image files from the folder, currently the image file means filenames ended with ".jpg" and ".png".
	 * @param folder
	 * @return ArrayList of Paths of images.
	 */
	static ArrayList<Path> CollectImageFiles(Path folder)
	{
		ArrayList<Path> imagePaths = new ArrayList<Path>();
		try (DirectoryStream<Path> dirStrm = Files.newDirectoryStream(folder)) 
		{			
			for (Path path : dirStrm)
			{
				String filename = path.toString();
				if (!filename.endsWith(".jpg") && !filename.endsWith(".png")) continue;
				
				imagePaths.add(path);
			}
		}
		catch (IOException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return imagePaths;
	}

	/**
	 * Collect XML files from the folder, currently the XML file means the filenames ended with ".xml".
	 * @param folder
	 * @return ArrayList of Paths of images.
	 */
	static ArrayList<Path> CollectXmlFiles(Path folder)
	{
		ArrayList<Path> xmlPaths = new ArrayList<Path>();
		try (DirectoryStream<Path> dirStrm = Files.newDirectoryStream(folder)) 
		{			
			for (Path path : dirStrm)
			{
				String filename = path.toString();
				if (!filename.endsWith(".xml")) continue;
				
				xmlPaths.add(path);
			}
		}
		catch (IOException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return xmlPaths;
	}

	/**
	 * Collect all image files from the folder, currently the image file means filenames ended with ".jpg" and ".png".
	 * @param folder
	 * @return ArrayList of Paths of images.
	 */
	static ArrayList<Path> CollectSubfolders(Path folder)
	{
		ArrayList<Path> folderPaths = new ArrayList<Path>();
		try (DirectoryStream<Path> dirStrm = Files.newDirectoryStream(folder)) 
		{			
			for (Path path : dirStrm)
			{
				if (path.toFile().isDirectory())
					folderPaths.add(path);
			}
		}
		catch (IOException e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return folderPaths;
	}
	
}


class RectangleTopAscending implements Comparator<Rectangle>
{
	public int compare(Rectangle o1, Rectangle o2) 
	{
		double diff = o1.y - o2.y;
		if (diff > 0) return 1;
		else if (diff == 0) return 0;
		else return -1;
	}
}

