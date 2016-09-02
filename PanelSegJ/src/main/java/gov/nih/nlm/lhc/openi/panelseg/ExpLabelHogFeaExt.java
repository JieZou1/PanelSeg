package gov.nih.nlm.lhc.openi.panelseg;

/**
 * Created by jzou on 8/31/2016.
 */
public class ExpLabelHogFeaExt extends Exp
{
    public static void main(String args[]) throws Exception {
        //Stop and print error msg if no arguments passed.
        if (args.length != 2) {
            System.out.println("Usage: java -cp PanelSegJ.jar ExpLabelHogFeaExt <Sample List File> <target folder>");
            System.out.println("	This is a utility program to do Label HoG feature extractions..");
            System.exit(0);
        }

        ExpLabelHogFeaExt feaExt = new ExpLabelHogFeaExt(args[0], args[1]);
        feaExt.featureExtract();
        System.out.println("Completed!");
    }

    /**
     * Ctor, set targetFolder and then collect all imagePaths
     * It also clean the targetFolder
     *
     * @param trainListFile
     * @param targetFolder
     */
    ExpLabelHogFeaExt(String trainListFile, String targetFolder) {
        super(trainListFile, targetFolder);
    }

    private void featureExtract()
    {
    }

}
