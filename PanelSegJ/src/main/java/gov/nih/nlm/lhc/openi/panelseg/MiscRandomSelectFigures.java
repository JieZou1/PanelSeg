package gov.nih.nlm.lhc.openi.panelseg;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.List;

/**
 * I have collected a large number of figure images in D:\Users\jie\projects\PanelSeg\data\downloads\jpgs folder.
 * This function is to randomly select some from that collection for generating another set of annotated samples.
 *
 * Created by jzou on 1/9/2017.
 */
public class MiscRandomSelectFigures
{
    public static void main(String args[]) throws Exception
    {
        if(args.length != 1)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar MiscRandomSelectFigures <number of samples>");
            System.out.println("	This is a utility program to randomly select figure images from D:\\Users\\jie\\projects\\PanelSeg\\data\\downloads\\jpgs folders");
            System.out.println("	The selected figure images are going to be moved to D:\\Users\\jie\\projects\\PanelSeg\\data\\downloads folder.");
            System.exit(0);
        }

        String foldersrc = "D:\\Users\\jie\\projects\\PanelSeg\\data\\downloads\\jpgs";
        String folderdst = "D:\\Users\\jie\\projects\\PanelSeg\\data\\downloads";

        List<Path> images = AlgMiscEx.collectImageFiles(Paths.get(foldersrc));

        List<Path> selected = AlgMiscEx.randomItems(images, 1500);

        for (Path src : selected)
        {
            Path dst = Paths.get(folderdst, src.getFileName().toString());

            Files.move(src, dst, StandardCopyOption.REPLACE_EXISTING);
        }
    }

}
