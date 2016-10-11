package gov.nih.nlm.lhc.openi.panelseg;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
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
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Files;
import java.util.Date;
import java.util.Random;

/**
 * LeNet method for Label Recognition
 *
 * Created by jzou on 10/3/2016.
 */
public class ExpLabelClassifyLeNet
{
    protected static final Logger log = LoggerFactory.getLogger(ExpLabelClassifyLeNet.class);

    //Images are of format given by allowedExtension -
    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    protected static final long seed = 12345;
    public static final Random randNumGen = new Random(seed);

    protected static int height = 28;
    protected static int width = 28;
    protected static int channels = 1;
    protected static int outputNum = 2;
    protected static int batchSize = 64;
    protected static int nEpochs = 10;
    protected static int iterations = 1;

    public static void main(String[] args) throws Exception
    {
        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.01)//.biasLearningRate(0.02)
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
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation("identity")
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
                .setInputType(InputType.convolutional(28,28,1)) //See note below
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
        File parentDir = new File("D:\\Users\\jie\\projects\\PanelSeg\\Exp\\LabelHog\\Classification2Class");

        //Files in directories under the parent dir that have "allowed extensions"
        // split needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        long total = filesInDir.locations().length;

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

        DataSetIterator trainDataIter, testDataIter;
        ImageRecordReader trainRecordReader = new ImageRecordReader(height,width,channels,labelMaker);
        trainRecordReader.initialize(trainData);
        ImageRecordReader testRecordReader = new ImageRecordReader(height,width,channels,labelMaker);
        testRecordReader.initialize(testData);

        log.info("Normalize data....");
        trainDataIter = new RecordReaderDataSetIterator(trainRecordReader, (int)totalTrain,1,outputNum);
        DataSet datasetAll = trainDataIter.next();
        //NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler();
        ImagePreProcessingScaler myScaler = new ImagePreProcessingScaler();
        myScaler.fit(datasetAll);
//        log.info("Min: {}",myScaler.getMin());
//        log.info("Max: {}",myScaler.getMax());

        trainDataIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1, outputNum);
        testDataIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1, outputNum);
        trainDataIter.setPreProcessor(myScaler);
        testDataIter.setPreProcessor(myScaler);

        log.info("Training Starts at: {}", (new Date()).toString());
        long startTime = System.nanoTime();

        model.setListeners(new ScoreIterationListener(10));
        for( int i=0; i<nEpochs; i++ )
        {
            model.fit(trainDataIter);
            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while(testDataIter.hasNext()){
                DataSet ds = testDataIter.next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
            }
            log.info(eval.stats());
            testDataIter.reset();
        }

        long endTime = System.nanoTime();
        log.info("Training Ends at: {}", (new Date()).toString());
        log.info("Training Time in Seconds: {}", (endTime - startTime)/ 1000000000.0);

        log.info("Save Model...");
        String modelFile = "LeNet5.model";
        ModelSerializer.writeModel(model, modelFile, true);

        log.info("Load Model...");
        MultiLayerNetwork model1 = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        log.info("Evaluate saved and then loaded model....");
        Evaluation eval = new Evaluation(outputNum);
        while(testDataIter.hasNext()){
            DataSet ds = testDataIter.next();
            INDArray output = model.output(ds.getFeatureMatrix(), false);
            eval.eval(ds.getLabels(), output);
        }
        log.info(eval.stats());
        testDataIter.reset();

        log.info("****************Completed********************");
    }

}
