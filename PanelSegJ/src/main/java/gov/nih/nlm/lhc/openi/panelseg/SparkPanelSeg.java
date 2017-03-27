package gov.nih.nlm.lhc.openi.panelseg;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

/**
 * Panel Segmentation method in Spark
 *
 * Created by jzou on 3/24/2017.
 */
public class SparkPanelSeg
{
    public static void main(String[] args) throws Exception
    {
        if (args.length != 2)
        {
            System.out.println("Usage: SparkPanelSeg <input path> <output path");
            System.exit(-1);
        }

        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("");
        sparkConf.setAppName("PanelSeg");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        JavaRDD<String> lines = sc.textFile(args[0]);
    }
}
