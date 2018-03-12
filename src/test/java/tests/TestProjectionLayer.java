package tests;

import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.ProjectionLayer.ProjectionLayer;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.ProjectionLayer.ProjectionLayerImpl;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestProjectionLayer {
    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;
    private static double alpha;

    @BeforeClass
    public static void before() {
        Config.readFile("./config.json");
        Nd4j.zeros(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        alpha = Config.getDouble("wordAlpha");
    }

    @Test
    public void testCalculations() {
        final ProjectionLayerImpl projectionLayer = new ProjectionLayerImpl(new NeuralNetConfiguration());

        final List<double[]> inputs = new ArrayList<>();

        inputs.add(new double[]{
                0, Math.pow(alpha, 0),
                245, Math.pow(alpha, 1),
                65432, Math.pow(alpha, 2),
                123, Math.pow(alpha, 3),
                -1,
                -1
        });

        inputs.add(new double[]{
                2, Math.pow(alpha, 0),
                1, Math.pow(alpha, 1),
                6437, Math.pow(alpha, 2),
                2, Math.pow(alpha, 3),
                -1,
                -1
        });

        final List<INDArray> input = inputs.stream().map(Nd4j::create).collect(Collectors.toList());
        projectionLayer.setInput(Nd4j.vstack(input));

        final int[] intIndices = projectionLayer.getIntIndices();

        assertArrayEquals(new int[]{0, 245, 65432, 123, 2, 1, 6437}, intIndices);

        final INDArray denseInput = projectionLayer.getDenseInput(intIndices);

        final INDArray expectedOutput = Nd4j.create(2, 7);
        expectedOutput.putScalar(0, 0, Math.pow(alpha, 0));
        expectedOutput.putScalar(0, 1, Math.pow(alpha, 1));
        expectedOutput.putScalar(0, 2, Math.pow(alpha, 2));
        expectedOutput.putScalar(0, 3, Math.pow(alpha, 3));

        expectedOutput.putScalar(1, 4, Math.pow(alpha, 0) + Math.pow(alpha, 3));
        expectedOutput.putScalar(1, 5, Math.pow(alpha, 1));
        expectedOutput.putScalar(1, 6, Math.pow(alpha, 2));

        assertEquals(expectedOutput, denseInput);
    }

    @Test
    public void testProjectionLayerSimple() {
        final Random r = new Random(12345);
        final int nExamples = 5;
        final int inputSize = 6;

        final INDArray input = Nd4j.valueArrayOf(nExamples, inputSize * 2, -1);
        final INDArray labels = Nd4j.create(nExamples, 3);

        for (int i = 0; i < nExamples; i++) {
            input.putScalar(i, r.nextInt(4));
            input.putScalar(i + 1, r.nextInt(4));
            input.putScalar(i + 2, r.nextInt(4));
            input.putScalar(i + 3, r.nextInt(4));
            input.putScalar(i + 4, r.nextInt(4));
            input.putScalar(i + 5, r.nextInt(4));

            labels.putScalar(new int[]{i, r.nextInt(3)}, 1.0);
        }

        final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .l2(0.2)
                .l1(0.1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .seed(12345L)
                .list()
                .layer(0, new ProjectionLayer.Builder()
                        .nIn(inputSize)
                        .nOut(3)
                        .weightInit(WeightInit.XAVIER)
                        .dist(new NormalDistribution(0, 1))
                        .updater(new NoOp())
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(3)
                        .nOut(3)
                        .weightInit(WeightInit.XAVIER)
                        .dist(new NormalDistribution(0, 1))
                        .updater(new NoOp())
                        .activation(Activation.SOFTMAX)
                        .build())
                .pretrain(false)
                .backprop(true)
                .build();

        final MultiLayerNetwork mln = new MultiLayerNetwork(conf);
        mln.init();

        if (PRINT_RESULTS) {
            System.out.println("testProjectionLayerSimple");
            for (int j = 0; j < mln.getnLayers(); j++)
                System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
        }

        final boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

        final String msg = "testProjectionLayerSimple";

        assertTrue(msg, gradOK);
    }

}
