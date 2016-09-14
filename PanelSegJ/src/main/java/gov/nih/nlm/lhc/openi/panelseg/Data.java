package gov.nih.nlm.lhc.openi.panelseg;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * The base class for all related operations on one set in data folder. <p>
 * This class is not intended to be instantiated, so we make it abstract.
 *
 * Created by jzou on 8/26/2016.
 */
abstract class Data
{
    Path setFolder;        //Data folder of a set
    ArrayList<Path> imagePaths;   //The imageColor file path of the set
    Path stylePath;               //The style file path
    Map<String, String> styles;   //The styles of the figures

    /**
     * Ctor, set setFolder and then collect all imagePaths
     * It also load the style annotation into styles.
     * If style.txt is not found in the setFolder, styles is set to null.
     * @param setFolder
     */
    Data(String setFolder)
    {
        this.setFolder = Paths.get(setFolder);
        imagePaths = AlgMiscEx.collectImageFiles(this.setFolder);
        System.out.println("Total number of imageColor is: " + imagePaths.size());

        stylePath = Paths.get(setFolder, "style.txt");
        styles = Data.loadStyleMap(stylePath);
    }

    /**
     * Read the style annotation from the file.
     * If the styleFile does not exist, return null.
     *
     * @param stylePath The filepath
     * @return imageColor file name string and style annotation in HashMap<string, string>
     */
    static HashMap<String, String> loadStyleMap(Path stylePath)
    {
        HashMap<String, String> styles = new HashMap<>();

        if(!Files.exists(stylePath) || Files.isDirectory(stylePath))
        {	//No styles have been marked yet
            System.out.println("Not able to find style.txt!");
            return null;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(stylePath.toFile())))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                String[] words = line.split("\\s+");
                styles.put(words[0], words[1]);
            }
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return styles;
    }

    /**
     * Save the style annotation into the file
     * @param stylePath
     * @param styles
     */
    static void saveStyleMap(Path stylePath, Map<String, String> styles)
    {
        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(stylePath.toFile()))))
        {
            for (Map.Entry<String, String> entry : styles.entrySet())
            {
                String key = entry.getKey();
                String value = entry.getValue();
                bw.write(key + "\t " + value);
                bw.newLine();
            }
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }


}
