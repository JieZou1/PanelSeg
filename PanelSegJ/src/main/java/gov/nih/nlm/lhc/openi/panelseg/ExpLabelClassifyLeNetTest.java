package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Some testing codes to try out
 *
 * Created by jzou on 10/26/2016.
 */
public class ExpLabelClassifyLeNetTest
{
    protected static final Logger log = LoggerFactory.getLogger(ExpLabelClassifyLeNetTrain.class);

    private static String testImageFolder = "\\Users\\jie\\projects\\PanelSeg\\Exp\\LabelHog\\Classification2Class\\testset\\";
    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    protected static int height = 28;
    protected static int width = 28;
    protected static int channels = 1;

    private static MultiLayerNetwork model;

    private static void loadModel() throws Exception
    {
        log.info("Load Model...");
        String modelFile = "LeNet5.model";
        model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
    }

    public static void testWithMats()  throws Exception
    {
        PrintWriter outputFile = new PrintWriter(new FileWriter("OutputWithMat.txt"));

        String[] subfolder = new String[] {"neg", "pos"};

        for (int k = 0; k < subfolder.length; k++)
        {   log.info("Test " + subfolder[k] + " images...");

            File[] testImageFiles = (new File(testImageFolder + subfolder[k])).listFiles(new FilenameFilter() {
                public boolean accept(File dir, String name) {
                    return name.toLowerCase().endsWith(".png");
                }
            });
            opencv_core.Mat[] images = new opencv_core.Mat[testImageFiles.length];
            for (int i = 0; i < testImageFiles.length; i++) {
                String testImageFile = testImageFiles[i].getAbsolutePath();
                images[i] = opencv_imgcodecs.imread(testImageFile);
            }

            log.info("Construct Mats to INDArrays...");
            NativeImageLoader imageLoader = new NativeImageLoader(width, height, channels);
            List<INDArray> slices = new ArrayList<>();
            for (int i = 0; i < images.length; i++) {
                INDArray arr = imageLoader.asMatrix(images[i]);
                slices.add(arr);
            }
            INDArray imageSet = new NDArray(slices, new int[]{images.length, channels, width, height});
            imageSet.divi(255.0);

            log.info("Predict...");
            INDArray results = model.output(imageSet);
            for (int i = 0; i < results.rows(); i++)
            {
                outputFile.println(results.getFloat(i, 0) + "\t" + results.getFloat(i, 1) + "\t" + testImageFiles[i].getName());
            }

            outputFile.println();
        }

        outputFile.close();
    }

    public static void testWithImageRecordReader() throws Exception
    {
        PrintWriter outputFile = new PrintWriter(new FileWriter("OutputWithImageRecordReader.txt"));

        FileSplit filesInDir = new FileSplit(new File(testImageFolder), allowedExtensions);

        log.info("Total files: {}", filesInDir.locations().length);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); //We have to use labelMaker, otherwise testDataIter.next() will raise null exception
        ImageRecordReader testRecordReader = new ImageRecordReader(height, width, channels, labelMaker);
        testRecordReader.initialize(filesInDir);

        ImagePreProcessingScaler myScaler = new ImagePreProcessingScaler(0, 1);
        DataSetIterator testDataIter = new RecordReaderDataSetIterator(testRecordReader, 64, 1, 2);
        testDataIter.setPreProcessor(myScaler);

        boolean hasNext = testDataIter.hasNext();
        DataSet ds = testDataIter.next();
        INDArray featureMatrix = ds.getFeatureMatrix();
        INDArray results = model.output(featureMatrix, false);

        for (int i = 0; i < results.rows(); i++)
        {
            outputFile.println(results.getFloat(i, 0) + "\t" + results.getFloat(i, 1) + "\t" + filesInDir.locations()[i].getPath());
        }

        outputFile.close();
    }

    public static void main(String[] args) throws Exception
    {
        loadModel();

        testWithMats();

        testWithImageRecordReader();

    }

}