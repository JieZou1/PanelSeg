package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;

import java.awt.*;
import java.awt.image.BufferedImage;

import static org.bytedeco.javacpp.opencv_core.CV_8UC1;
import static org.bytedeco.javacpp.opencv_core.CV_8UC3;
import static org.bytedeco.javacpp.opencv_core.findNonZero;
import static org.bytedeco.javacpp.opencv_imgproc.boundingRect;

/**
 * Created by jzou on 8/31/2016.
 */
final class AlgOpenCVEx
{
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
     * @return Mat imageColor in BGR format
     */
    static opencv_core.Mat bufferdImg2Mat(BufferedImage in)
    {
        if (in.getType() != BufferedImage.TYPE_INT_RGB)		in = convert(in, BufferedImage.TYPE_INT_RGB);

        opencv_core.Mat out;
        byte[] data;         int r, g, b;          int height = in.getHeight(), width = in.getWidth();
        if(in.getType() == BufferedImage.TYPE_INT_RGB || in.getType() == BufferedImage.TYPE_INT_ARGB)
        {
            out = new opencv_core.Mat(height, width, CV_8UC3);
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
            out = new opencv_core.Mat(height, width,  CV_8UC1);
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
     * Convert Mat imageColor to BufferedImage <p>
     * BufferedImage is either in TYPE_BYTE_GRAY or TYPE_INT_RGB format.
     * @param in
     * @return
     */
    static BufferedImage mat2BufferdImg(opencv_core.Mat in)
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
//			ImageIO.write(out, "jpg", new File("temp.jpg"));
        return out;
    }

    /**
     * Crop the ROI from the imageColor
     * @param image
     * @param roi
     * @return
     */
    static opencv_core.Mat cropImage(opencv_core.Mat image, Rectangle roi)
    {
        opencv_core.Rect rect = new opencv_core.Rect(roi.x, roi.y, roi.width, roi.height);
        return image.apply(rect);
    }

    /**
     * Convert java.awt.Rectangle to opencv_core.Rect
     * @param rectangle
     * @return
     */
    static opencv_core.Rect Rectangle2Rect(Rectangle rectangle)
    {
        return new opencv_core.Rect(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
    }

    /**
     * Get a color, for drawing on Mat (imageColor)
     * @param i
     * @return
     */
    static opencv_core.Scalar getColor(int i)
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
        return new opencv_core.Scalar(color.getBlue(), color.getGreen(), color.getRed(), 0);
    }

    /**
     * Find the bounding box of a binary imageColor. White as foreground
     * @param bina
     * @return
     */
    static opencv_core.Rect findBoundingbox(opencv_core.Mat bina)
    {
        opencv_core.Mat points = new opencv_core.Mat();	findNonZero(bina,points);
        return boundingRect(points);
    }


}
