package gov.nih.nlm.lhc.openi.panelseg;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Properties;

/**
 * Created by jzou on 2/8/2017.
 */
public class ExpPanelSeg1 extends Exp
{
    protected static final Logger log = LoggerFactory.getLogger(ExpPanelSeg1.class);

    public static void main(String args[])
    {
        ExpPanelSeg1 expPanelSeg = new ExpPanelSeg1();
        try
        {
            expPanelSeg.loadProperties();
        }
        catch (Exception ex)
        {
            log.error(ex.getMessage());
        }
    }

    private PanelSeg.Method method;

    private ExpPanelSeg1() {super();}

    @Override
    void initialize(String propertyFile) throws Exception {

    }

    @Override
    void doWork() throws Exception {

    }

    /**
     * Load the properties from ExpPanelSeg.properties file.
     * Also, validate all property values, throw exceptions if not valid.
     * @throws Exception
     */
    void loadProperties() throws Exception
    {
        //Load properties
        properties = new Properties();
        properties.load(this.getClass().getClassLoader().getResourceAsStream("ExpPanelSeg.properties"));

        String strListFile = properties.getProperty("listFile");
        if (strListFile == null) throw new Exception("ERROR: listFile property is Missing.");
        File list_file = new File(strListFile);
        if (!list_file.exists()) throw new Exception("ERROR: " + list_file + " does not exist.");
        if (!list_file.isFile()) throw new Exception("ERROR: " + list_file + " is not a file.");
        listFile = list_file.toPath();

        String strTargetFolder = properties.getProperty("targetFolder");
        if (strTargetFolder == null) throw new Exception("ERROR: targetFolder property is Missing.");
        File target_folder = new File(strTargetFolder);
        targetFolder = target_folder.toPath();
        AlgMiscEx.createClearFolder(this.targetFolder);

        String strMethod = properties.getProperty("method");
        if (strMethod == null) throw new Exception("ERROR: method property is Missing.");
        switch (strMethod)
        {
            case "LabelDetHog":  method = PanelSeg.Method.LabelDetHog; break;
            case "LabelRegHogSvm": method = PanelSeg.Method.LabelRegHogSvm; break;
            case "LabelRegHogSvmThreshold": method = PanelSeg.Method.LabelRegHogSvmThreshold; break;
            case "LabelRegHogSvmBeam": method = PanelSeg.Method.LabelRegHogSvmBeam; break;

            case "LabelDetHogLeNet5": method = PanelSeg.Method.LabelDetHogLeNet5; break;
            case "LabelRegHogLeNet5Svm": method = PanelSeg.Method.LabelRegHogLeNet5Svm; break;
            case "LabelRegHogLeNet5SvmBeam": method = PanelSeg.Method.LabelRegHogLeNet5SvmBeam; break;
            case "LabelRegHogLeNet5SvmAlignment": method = PanelSeg.Method.LabelRegHogLeNet5SvmAlignment; break;

            case "PanelSplitSantosh": method = PanelSeg.Method.PanelSplitSantosh; break;
            case "PanelSplitJaylene": method = PanelSeg.Method.PanelSplitJaylene; break;
            default: throw new Exception(strMethod + " is Unknown");
        }

        PanelSeg.initialize(method, properties);
    }


    @Override
    void doWork(int k) throws Exception {

    }
}
