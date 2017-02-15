package gov.nih.nlm.lhc.openi.panelseg;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.flow.FlowIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;

/**
 * Created by jzou on 2/14/2017.
 */
final class ExpTrainLabelClassifyLeNet5 extends Exp
{
    public static void main(String args[])
    {
        log.info("Training tasks for LeNet5 Label Classification.");

        ExpTrainLabelClassifyLeNet5 exp = new ExpTrainLabelClassifyLeNet5();
        try
        {
            exp.loadProperties();
            exp.initialize();
            exp.doWork();
            log.info("Completed!");
        }
        catch (Exception ex)
        {
            log.error(ex.getMessage());
        }
    }

    private enum Task { Train, Test }
    private Task task;
    private String propEpoch, propLabelLeNet5Model;

    @Override
    void loadProperties() throws Exception
    {
        loadProperties("ExpTrainLabelClassifyLeNet5.properties");

        String strTask = getProperty("Task");
        switch (strTask) {
            case "Train":
                task = Task.Train;
                loadPropertiesTrain();
                break;
            case "Test":
                task = Task.Test;
                loadPropertiesTest();
                break;
            default: throw new Exception("Task " + strTask + " is Unknown");
        }

        waitKeyContinueOrQuit("Configuration Okay? Press any key to continue, press N to quit");
    }

    private void loadPropertiesTrain() throws Exception
    {
        propTrainFolder = getProperty("TrainFolder");
        propTargetFile = getProperty("TargetFile");
        propEpoch = getProperty("Epoch");
    }

    private void loadPropertiesTest() throws Exception
    {
        propTestFolder = getProperty("TestFolder");
        propLabelLeNet5Model = getProperty("LabelLeNet5Model");
    }

    @Override
    void initialize() throws Exception
    {
        switch (task)
        {
            case Train: initializeTrain(); break;
            case Test: initializeTest(); break;
        }
    }

    private void initializeTrain() throws Exception
    {
        setTrainFolder();
    }

    private void initializeTest() throws Exception
    {
        setTestFolder();
    }

    @Override
    void doWork() throws Exception {
        switch (task)
        {
            case Train: doWorkLeNet5Train(); break;
            case Test: doWorkLeNet5Test(); break;
        }

    }
    private void doWorkLeNet5Train() throws Exception
    {
        String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        long seed = 12345;
        Random randNumGen = new Random(seed);

        int height = 28;
        int width = 28;
        int channels = 1;
        int outputNum = 2;
        int batchSize = 64;
        int iterations = 1;
        int nEpochs = Integer.parseInt(propEpoch);

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations) //Usually use 1, indicating that 1 parameter update for 1 minibatch
                .regularization(true)./*dropOut(0.5)//.*/l2(0.0005)    //typical value is 0.001 to 0.000001
                .learningRate(0.005)//.biasLearningRate(0.02) //typical value is 0.1 to 0.000001
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        //.activation("identity")
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        //.activation("identity")
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .setInputType(InputType.convolutional(width,height,1)) //See note below
                .backprop(true).pretrain(false);

        /*
        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
            and the dense layers
        (b) Does some additional configuration validation
        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
            layer based on the size of the previous layer (but it won't override values manually set by the user)

        In earlier versions of DL4J, the (now deprecated) ConvolutionLayerSetup class was used instead for this.
        InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
        For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
        MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
        row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
         */

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Load data....");
        File parentDir = trainFolder.toFile();

        //Files in directories under the parent dir that have "allowed extensions"
        // split needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        log.info("Total files: {}", filesInDir.locations().length);

        //You do not have to manually specify labels. This class (instantiated as below) will
        //parse the parent dir and use the name of the subdirectories as label/class names
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        //Below is a bare bones version. Refer to javadocs for details
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        long totalTrain = trainData.locations().length;
        long totalTest = testData.locations().length;

        log.info("Total train files: {}", totalTrain);
        log.info("Total test files: {}", totalTest);

        log.info("Normalize data....");

        ImageRecordReader trainRecordReader = new ImageRecordReader(height,width,channels,labelMaker);
        trainRecordReader.initialize(trainData);
        ImageRecordReader testRecordReader = new ImageRecordReader(height,width,channels,labelMaker);
        testRecordReader.initialize(testData);

        ImagePreProcessingScaler myScaler = new ImagePreProcessingScaler(0, 1);

        DataSetIterator trainDataIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1, outputNum);
        DataSetIterator testDataIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1, outputNum);
        trainDataIter.setPreProcessor(myScaler);
        testDataIter.setPreProcessor(myScaler);

        log.info("Training Starts at: {}", (new Date()).toString());
        long startTime = System.nanoTime();

        model.setListeners(new ScoreIterationListener(10));
        //model.setListeners(new HistogramIterationListener(1));
        //model.setListeners(new FlowIterationListener(10));
        for( int i=0; i<nEpochs; i++ )
        {
            model.fit(trainDataIter);
            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while(testDataIter.hasNext()){
                DataSet ds = testDataIter.next();
                INDArray featureMatrix = ds.getFeatureMatrix();
                INDArray output = model.output(featureMatrix, false);
                eval.eval(ds.getLabels(), output);
            }
            log.info(eval.stats());
            testDataIter.reset();
        }

        long endTime = System.nanoTime();
        log.info("Training Ends at: {}", (new Date()).toString());
        log.info("Training Time in Seconds: {}", (endTime - startTime)/ 1000000000.0);

        log.info("Save Model...");
        String modelFile = propTargetFile;
        ModelSerializer.writeModel(model, modelFile, true);

        log.info("Load Model...");
        MultiLayerNetwork model1 = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        log.info("Evaluate saved and then loaded model....");
        Evaluation eval = new Evaluation(outputNum);
        while(testDataIter.hasNext()){
            DataSet ds = testDataIter.next();
            INDArray output = model1.output(ds.getFeatureMatrix(), false);
            eval.eval(ds.getLabels(), output);
        }
        log.info(eval.stats());
        testDataIter.reset();

        log.info("****************Completed********************");
    }

    private void doWorkLeNet5Test() throws Exception
    {
        int height = 28;
        int width = 28;
        int channels = 1;

        MultiLayerNetwork model = loadModel();
        testWithMats(model, width, height, channels);
        //testWithImageRecordReader(model, width, height, channels);
    }

    private MultiLayerNetwork loadModel() throws Exception
    {
        log.info("Load Model...");
        InputStream modelStream = LabelClassifyLeNet5.class.getClassLoader().getResourceAsStream(propLabelLeNet5Model);
        MultiLayerNetwork leNet5Model = ModelSerializer.restoreMultiLayerNetwork(modelStream);
        System.out.println(propLabelLeNet5Model + " is loaded.");
        return leNet5Model;
    }

    public void testWithMats(MultiLayerNetwork model, int width, int height, int channels)  throws Exception
    {
        PrintWriter outputFile = new PrintWriter(new FileWriter("OutputWithMat.txt"));

        String[] subfolder = new String[] {"neg", "pos"};

        for (int k = 0; k < subfolder.length; k++)
        {   log.info("Test " + subfolder[k] + " images...");

            File[] testImageFiles = testFolder.resolve(subfolder[k]).toFile().listFiles(new FilenameFilter() {
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

    public void testWithImageRecordReader(MultiLayerNetwork model, int width, int height, int channels) throws Exception
    {
        PrintWriter outputFile = new PrintWriter(new FileWriter("OutputWithImageRecordReader.txt"));

        FileSplit filesInDir = new FileSplit(testFolder.toFile(), BaseImageLoader.ALLOWED_FORMATS);

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

}
