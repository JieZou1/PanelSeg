package gov.nih.nlm.lhc.openi.panelseg;

/**
 * Bootstrap of HOG+SVM method for Label Detection
 * Collect negative and positive patches for bootstrapping.
 *
 * Created by jzou on 9/9/2016.
 */
final class ExpLabelDetectHogBootstrap extends Exp
{
    public static void main(String args[]) throws Exception {
        //Stop and print error msg if no arguments passed.
        if (args.length != 2) {
            System.out.println("Usage: java -cp PanelSegJ.jar ExpLabelDetectHogBootstrap <Sample List File> <target folder>");
            System.out.println("	This is a utility program to Collect negative and positive patches for bootstrapping.");
            System.exit(0);
        }

        ExpLabelDetectHogBootstrap generator = new ExpLabelDetectHogBootstrap(args[0], args[1]);
        generator.generate();
        System.out.println("Completed!");
    }

    /**
     * Ctor, set targetFolder and then collect all imagePaths
     * It also clean the targetFolder
     *
     * @param trainListFile
     * @param targetFolder
     */
    ExpLabelDetectHogBootstrap(String trainListFile, String targetFolder) {
        super(trainListFile, targetFolder, false);
    }

    /**
     * Entry function
     */
    void generate()
    {
    }
}
