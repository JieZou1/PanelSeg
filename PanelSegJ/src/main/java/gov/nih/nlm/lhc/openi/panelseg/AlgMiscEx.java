package gov.nih.nlm.lhc.openi.panelseg;

/**
 * Created by jzou on 8/31/2016.
 */

import org.apache.commons.io.FileUtils;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

/**
 * Created by jzou on 8/25/2016.
 */
final class AlgMiscEx
{
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
     * Collect all imageColor files from the folder, currently the imageColor file means filenames ended with ".jpg", ".bmp", and ".png".
     * @param folder
     * @return ArrayList of Paths of images.
     */
    static ArrayList<Path> collectImageFiles(Path folder)
    {
        ArrayList<Path> imagePaths = new ArrayList<>();
        try (DirectoryStream<Path> dirStrm = Files.newDirectoryStream(folder))
        {
            for (Path path : dirStrm)
            {
                String filename = path.toString();
                if (!filename.endsWith(".jpg") && !filename.endsWith(".png") && !filename.endsWith(".bmp")) continue;

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
    static ArrayList<Path> collectXmlFiles(Path folder)
    {
        ArrayList<Path> xmlPaths = new ArrayList<>();
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
     * Collect all sub-folders from the folder.
     * @param folder
     * @return ArrayList of Paths of images.
     */
    static ArrayList<Path> collectSubfolders(Path folder)
    {
        ArrayList<Path> folderPaths = new ArrayList<>();
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

    /**
     * If the folder doesn't exist, create it.
     * If it exists, clear it.
     * @param folder
     */
    static void createClearFolder(Path folder)
    {
        //Clean the folder
        if (!Files.exists(folder)) folder.toFile().mkdir();

        //Remove all file in preview folder
        try {
            FileUtils.cleanDirectory(folder.toFile());
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }
}
