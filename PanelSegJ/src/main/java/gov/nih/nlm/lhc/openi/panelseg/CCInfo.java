package gov.nih.nlm.lhc.openi.panelseg;

import java.awt.*;

/**
 * Created by jzou on 4/18/2017.
 */
public class CCInfo
{
    CCInfo(int left, int top, int width, int height, int size)
    {
        this.rectangle = new Rectangle(left,top,width,height);
        this.size = size;
    }
    Rectangle rectangle;
    int size;
}
