package gov.nih.nlm.lhc.openi.panelseg;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Created by jzou on 8/25/2016.
 */
class MiscCopyOriginalImage
{
    public static void main(String args[]) throws Exception
    {
        //Stop and print error msg if no arguments passed.
        if(args.length != 2)
        {
            System.out.println("Usage: java -cp PanelSegJ.jar MiscCopyOriginalImage <src imgLarge list> <dst imgOrignal folder>");
            System.out.println("	This is a utility program to copy original figure images from /hadoop/scratch folder on lhce-zol2 server.");
            System.out.println("	This program can run on lhce-zol2 server only.");
            System.out.println("	<src imgLarge list> is a text file, containing a list of figure images that we want to get.");
            System.out.println("	<dst imgOriginal folder> is a folder, where we want to save the retrieved original figure images.");
            System.out.println("	The program looks for the following mapping files:");
            System.out.println("	PMC: /hadoop/scratch/openi-v3.0/input-lists/feature-extraction/openi-v3.0-figures.txt");
            System.out.println("	HMD imageColor list: /hadoop/scratch/hmd/input-lists/feature-extraction/hmd-feature-extraction-list.txt");
            System.out.println("	CXR: /hadoop/storage/radiology/input-lists/feature-extraction/radiology-images.txt");
            System.out.println("	USC: /hadoop/storage/allcopied-usc/input-lists/feature-extraction/usc-imageColor-list.txt");
            System.exit(0);
        }

        MiscCopyOriginalImage copy = new MiscCopyOriginalImage(args[0], args[1]);
        copy.map(); //Mapping to find original file location from mapping
        copy.copy();
    }

    private Map<String, String> mapping;
    private String dstFolder;
    private ArrayList<String> srcList, dstList;

    /**
     * ctor, read in key-value pair mapping, src imgLarge list, and set dstFolder
     *
     * @param src_list_file
     * @param dst_folder
     */
    private MiscCopyOriginalImage(String src_list_file, String dst_folder) throws Exception
    {
        dstFolder = dst_folder;

        String[] mapping_files = new String[] {
                "/hadoop/scratch/openi-v3.0/input-lists/feature-extraction/openi-v3.0-figures.txt", //PMC
                "/hadoop/scratch/hmd/input-lists/feature-extraction/hmd-feature-extraction-list.txt", //HMD imageColor list"
                "/hadoop/storage/radiology/input-lists/feature-extraction/radiology-images.txt", //CXR
                "/hadoop/storage/usc/input-lists/feature-extraction/usc-imageColor-list.txt" //USC
        };

        //load mapping files
        mapping = new HashMap<>();
        for (int i = 0; i < mapping_files.length; i++)
        {
            String mapping_file = mapping_files[i];
            System.out.println("Loading " + mapping_file);
            try (BufferedReader br = new BufferedReader(new FileReader(mapping_file)))
            {
                String line;
                while ((line = br.readLine()) != null)
                {
                    String[] words = line.split("##");
                    mapping.put(words[0], words[1]);
                }
            }
        }
        System.out.println(Integer.toString(mapping.size()) + " key-value pairs are loaded.");

        //load src list
        System.out.println("Loading " + src_list_file);
        srcList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(src_list_file)))
        {
            String line;
            while ((line = br.readLine()) != null)
            {
                srcList.add(line.trim());
            }
        }
        System.out.println(Integer.toString(srcList.size()) + " figures in " + src_list_file);
    }

    private void map()
    {
        dstList = new ArrayList<>();
        for (int i = 0; i < srcList.size(); i++)
        {
            String src = srcList.get(i);
            src = src.replace(".png", ".jpg");

            String dst = mapping.get(src);
            if (dst == null)
                System.out.println(dst + " can not be found in the mapping key-value pair list");
            else
                dstList.add(dst);
        }
        System.out.println(Integer.toString(dstList.size()) + " figures to be copied.");
    }

    private void copy() throws IOException
    {
        //check whether the file exists
        //for (int i = 0; i < 5; i++)
        for (int i = 0; i < dstList.size(); i++)
        {
            Path src = Paths.get(dstList.get(i));
            Path dst = Paths.get(dstFolder, src.getFileName().toString());

            if (!Files.exists(src) || Files.isDirectory(src))
            {
                System.out.println(src.toString() + " does not exist.");
            }
            else
            {
                Files.copy(src, dst, StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }


}
