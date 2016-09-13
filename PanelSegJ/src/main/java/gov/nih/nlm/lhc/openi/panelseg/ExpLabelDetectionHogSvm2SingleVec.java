package gov.nih.nlm.lhc.openi.panelseg;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Convert Linear SVM model to Single Vector for HOG+SVM label detection
 *
 * Created by jzou on 9/9/2016.
 */
final class ExpLabelDetectionHogSvm2SingleVec
{
    public static void main(String args[]) throws Exception {
        //Stop and print error msg if no arguments passed.
        if (args.length != 1) {
            System.out.println("Usage: java -cp PanelSegJ.jar ExpLabelDetectionHogSvm2SingleVec <target folder>");
            System.out.println("	This is a utility program to Convert Linear SVM model to Single Vector for HOG+SVM label detection.");
            System.exit(0);
        }

        ExpLabelDetectionHogSvm2SingleVec generator = new ExpLabelDetectionHogSvm2SingleVec(args[0]);
        generator.generate();
        System.out.println("Completed!");
    }

    private Path targetFolder;    //The folder for saving the result

    /**
     * Ctor, set targetFolder and then collect all imagePaths
     * It also clean the targetFolder
     *
     * @param targetFolder
     */
    ExpLabelDetectionHogSvm2SingleVec(String targetFolder) {
        this.targetFolder = Paths.get(targetFolder);
    }

    /**
     * Entry function
     */
    void generate()
    {
        Path vectorPath = targetFolder.resolve("vector.java");

        //Save to a java file
        try (PrintWriter pw = new PrintWriter(vectorPath.toString()))
        {
            pw.println("package gov.nih.nlm.lhc.openi.panelseg;");
            pw.println();

            for (String name : PanelSegLabelRegHog.labelSetsHOG)
            {
                Path folder = targetFolder.resolve(name);
                Path folderModel = folder.resolve("model");
                Path modelPath = folderModel.resolve("svm_model");

                float[] singleVector = LibSvmEx.ToSingleVector(modelPath.toString());

                String classname =	"PanelSegLabelRegHoGModel_" + name;

                pw.println("final class " + classname);
                pw.println("{");

                pw.println("	static float[] svmModel = ");

                pw.println("    	{");
                for (int k = 0; k < singleVector.length; k++)
                {
                    pw.print(singleVector[k] + "f,");
                }
                pw.println();
                pw.println("    };");
                pw.println("}");
            }
        }
        catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }
}
