package gov.nih.nlm.lhc.openi.panelseg;

/**
 * Created by jzou on 8/31/2016.
 */

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.util.CombinatoricsUtils;

import java.awt.*;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

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
    static ArrayList<Path> collectImageFiles(Path folder) throws Exception
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
        return imagePaths;
    }

    /**
     * Collect XML files from the folder, currently the XML file means the filenames ended with ".xml".
     * @param folder
     * @return ArrayList of Paths of images.
     */
    static ArrayList<Path> collectXmlFiles(Path folder) throws Exception
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
        return xmlPaths;
    }

    /**
     * Collect all sub-folders from the folder.
     * @param folder
     * @return ArrayList of Paths of images.
     */
    static ArrayList<Path> collectSubfolders(Path folder) throws Exception
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
        return folderPaths;
    }

    /**
     * If the folder doesn't exist, create it.
     * If it exists, clear it.
     * @param folder
     */
    static void createClearFolder(Path folder) throws Exception
    {
        //Clean the folder
        if (!Files.exists(folder)) folder.toFile().mkdir();
        FileUtils.cleanDirectory(folder.toFile());
    }

    /**
     * Select one item from an ArrayList
     * @param myList
     * @param <T>
     * @return
     */
    static <T> T randomItem( List<T> myList) {
        Random rand = new Random();
        T randomInt = myList.get(rand.nextInt(myList.size()));
        return randomInt;
    }

    /**
     * Select n element (no duplicates) from an ArrayList
     * @param list
     * @param n
     * @param <T>
     * @return
     */
    static <T> List<T> randomItems( List<T> list, int n)
    {
        Random rand = new Random();

        List<Integer> selectedIndexes = new ArrayList<>();
        while (true)
        {
            int index = rand.nextInt(list.size());
            if (selectedIndexes.indexOf(index) == -1)
            {
                selectedIndexes.add(index);
            }
            if (selectedIndexes.size() == n) break;
        }

        List<T> selected = new ArrayList<>();
        for (int i = 0; i < n; i++)
        {
            int index = selectedIndexes.get(i);
            selected.add(list.get(index));
        }

        return selected;
    }

    /**
     * Randomly select k sets of an n-item list, n >= k
     * No duplicate items in one set; and no duplicate sets
     * @param list
     * @param k
     * @param max
     * @param <T>
     * @return
     */
    static <T> List<List<T>> randomItemSets( List<T> list, int k, int max)
    {
        List<List<T>> selectedItemSets = new ArrayList<>();
        if (k == 0)            return null;

        int n = list.size();
        double F0 = CombinatoricsUtils.factorialDouble(n);
        double F1 = CombinatoricsUtils.factorialDouble(n-k);
        double F2 = CombinatoricsUtils.factorialDouble(k);
        int MAX = (int)(F0/(F1 * F2));

        if (Double.isFinite(F0) && MAX < 1000)
        {
            List<int[]> selectedIndexes = CombinationSelection.select(n, k);
            if (selectedIndexes.size() > max)
            {
                Collections.shuffle(selectedIndexes, new Random());
                selectedIndexes = selectedIndexes.subList(0, max);
            }

            for (int i = 0; i < selectedIndexes.size(); i++)
            {
                int[] indexes = selectedIndexes.get(i);
                List<T> itemSets = new ArrayList<>();
                for (int j = 0; j < indexes.length; j++)
                    itemSets.add(list.get(indexes[j]));
                selectedItemSets.add(itemSets);
            }
        }
        else
        {   //Too many, we have to randomly select to avoid out of memory
            List<Integer> indexes = new ArrayList<>();   for (int i = 0; i < n; i++) indexes.add(i);

            List<List<Integer>> selectedIndexes = new ArrayList<>();
            while (true)
            {
                List<Integer> selectedIndex = randomItems(indexes, k);
                if (indexof(selectedIndexes, selectedIndex) == -1)
                {
                    selectedIndexes.add(selectedIndex);
                }
                if (selectedIndexes.size() == max)    break;
            }
            for (int i = 0; i < selectedIndexes.size(); i++)
            {
                indexes = selectedIndexes.get(i);
                List<T> itemSets = new ArrayList<>();
                for (int j = 0; j < indexes.size(); j++)
                    itemSets.add(list.get(indexes.get(j)));
                selectedItemSets.add(itemSets);
            }
        }

        return selectedItemSets;
    }

    /**
     * Return true if index1 and index2 contain the same numbers; otherwise false
     * @param index1
     * @param index2
     * @return
     */
    static boolean equal(List<Integer> index1, List<Integer> index2)
    {
        for (int i = 0; i < index1.size(); i++)
        {
            int i1 = index1.get(i), i2 = index2.get(i);
            if (i1 != i2) return false;
        }
        return true;
    }

    /**
     * Return the index of an element in indexes, which is the same as index1; if not found, return -1
     * @param indexes
     * @param index1
     * @return
     */
    static int indexof(List<List<Integer>> indexes, List<Integer> index1)
    {
        for (int i = 0; i < indexes.size(); i++)
        {
            List<Integer> index2 = indexes.get(i);
            if (equal(index1, index2)) return i;
        }
        return -1;
    }

    static int maxOverlappingPanel(Rectangle rect, List<Panel> panels)
    {
        //find max overlapping
        int maxIndex = -1; double maxSize = -1;
        for (int j = 0; j < panels.size(); j++)
        {
            Panel panel = panels.get(j);
            Rectangle intersection = panel.panelRect.intersection(rect);
            if (intersection.isEmpty()) continue;
            double size = intersection.width * intersection.height;
            if (size > maxSize)
            {
                maxIndex = j; maxSize = size;
            }
        }
        return maxIndex;
    }

    static int maxOverlappingCC(Rectangle rect, List<CCInfo> ccs)
    {
        //find max overlapping
        int maxIndex = -1; double maxSize = -1;
        for (int j = 0; j < ccs.size(); j++)
        {
            CCInfo cc = ccs.get(j);
            Rectangle intersection = cc.rectangle.intersection(rect);
            if (intersection.isEmpty()) continue;
            double size = intersection.width * intersection.height;
            if (size > maxSize)
            {
                maxIndex = j; maxSize = size;
            }
        }
        return maxIndex;
    }

    static int closestPanel(Rectangle rect, List<Panel> panels)
    {
        int minIndex = -1; int minDistance = Integer.MAX_VALUE;
        for (int j = 0; j < panels.size(); j++)
        {
            Panel panel = panels.get(j);
            int horDistance, verDistance, distance;
            if (rect.x < panel.panelRect.x)
            {
                horDistance = panel.panelRect.x - (rect.x + rect.width);
                if (horDistance < 0) horDistance = 0;
            }
            else
            {
                horDistance = rect.x - (panel.panelRect.x + panel.panelRect.width);
                if (horDistance < 0) horDistance = 0;
            }
            if (rect.y < panel.panelRect.y)
            {
                verDistance = panel.panelRect.y - (rect.y + rect.height);
                if (verDistance < 0) verDistance = 0;
            }
            else
            {
                verDistance = rect.y - (panel.panelRect.y + panel.panelRect.height);
                if (verDistance < 0) verDistance = 0;
            }
            distance = verDistance + horDistance;
            if (distance < minDistance)
            {
                minDistance = distance; minIndex = j;
            }
        }
        return minIndex;
    }

    static int closestCC(Rectangle rect, List<CCInfo> ccs)
    {
        int minIndex = -1; int minDistance = Integer.MAX_VALUE;
        for (int j = 0; j < ccs.size(); j++)
        {
            CCInfo cc = ccs.get(j);
            int horDistance, verDistance, distance;
            if (rect.x < cc.rectangle.x)
            {
                horDistance = cc.rectangle.x - (rect.x + rect.width);
                if (horDistance < 0) horDistance = 0;
            }
            else
            {
                horDistance = rect.x - (cc.rectangle.x + cc.rectangle.width);
                if (horDistance < 0) horDistance = 0;
            }
            if (rect.y < cc.rectangle.y)
            {
                verDistance = cc.rectangle.y - (rect.y + rect.height);
                if (verDistance < 0) verDistance = 0;
            }
            else
            {
                verDistance = rect.y - (cc.rectangle.y + cc.rectangle.height);
                if (verDistance < 0) verDistance = 0;
            }
            distance = verDistance + horDistance;
            if (distance < minDistance)
            {
                minDistance = distance; minIndex = j;
            }
        }
        return minIndex;
    }
}


/**
 * A quick (not efficient) solution for selecting k elements from n element set, (C(k,n))
 */
final class CombinationSelection
{
    static ArrayList<int[]> select(int elementLength, int selectionLength)
    {
        CombinationSelection selection = new CombinationSelection(elementLength);
        return selection.select(selectionLength);
    }

    private int[] elements;
    private ArrayList<int[]> selection;

    private CombinationSelection(int elementLength)
    {
        elements = new int[elementLength];
        for (int i = 0; i < elementLength; i++) elements[i] = i;
    }

    private ArrayList<int[]> select(int selectionLength)
    {
        selection = new ArrayList<>();
        combinations2(elements, selectionLength, 0, new int[selectionLength] );

        return selection;
    }

    private void combinations2(int[] arr, int len, int startPosition, int[] result) {
        if (len == 0) {
            selection.add(result.clone());
            return;
        }
        for (int i = startPosition; i <= arr.length - len; i++) {
            result[result.length - len] = arr[i];
            combinations2(arr, len - 1, i + 1, result);
        }
    }
}
