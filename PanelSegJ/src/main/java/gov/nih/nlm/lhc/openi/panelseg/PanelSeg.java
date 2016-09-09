package gov.nih.nlm.lhc.openi.panelseg;

/**
 * The base class for all panel segmentation algorithms. <p>
 * This class is not intended to be instantiated, so we make it abstract.
 *
 * Created by jzou on 8/25/2016.
 */
public abstract class PanelSeg
{
    //All possible panel label chars, 'c', 'k', 'o', 'p', 's', 'u', 'v' 'w', 'x', 'y', 'z' no difference between upper and lower cases.
    static final char[] labelChars = {
            'a', 'A', 'b', 'B', 'c', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H',
            'i', 'I', 'j', 'J', 'k', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'p', 'q', 'Q',
            'r', 'R', 's', 't', 'T', 'u', 'v', 'w', 'x', 'y', 'z',
            '1', '2', '3', '4', '5', '6', '7', '8', '9'
    };

    /**
     * Convert label char to folder name. 'a' and 'A' are 2 different char, but a and A folders are the same.
     * @param labelChar
     * @return
     */
    static String getLabelCharFolderName(char labelChar) {
        //Special treatment for those identical upper and lower chars
        Character labelCharLower = Character.toLowerCase(labelChar);
        if (labelCharLower == 'c') return "c";
        if (labelCharLower == 'k') return "k";
        if (labelCharLower == 'o') return "o";
        if (labelCharLower == 'p') return "p";
        if (labelCharLower == 's') return "s";
        if (labelCharLower == 'u') return "u";
        if (labelCharLower == 'v') return "v";
        if (labelCharLower == 'w') return "w";
        if (labelCharLower == 'x') return "x";
        if (labelCharLower == 'y') return "y";
        if (labelCharLower == 'z') return "z";

        return Character.isUpperCase(labelChar) ? labelChar + "_" : Character.toString(labelChar);
    }
}
