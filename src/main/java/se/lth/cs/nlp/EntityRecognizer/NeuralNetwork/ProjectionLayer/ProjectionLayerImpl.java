package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.ProjectionLayer;

import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.custom.ScatterUpdate;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class ProjectionLayerImpl extends BaseLayer<ProjectionLayer> {
    private static final int[] SCATTER_DIM = new int[]{1};
    private int[] intIndices;
    private INDArray denseInput;

    public ProjectionLayerImpl(final NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public INDArray preOutput(final boolean training) {
        intIndices = getIntIndices();
        denseInput = getDenseInput(intIndices);

        final INDArray weights = getParam(DefaultParamInitializer.WEIGHT_KEY);
        final INDArray weightRows = Nd4j.pullRows(weights, 1, intIndices);

        return denseInput.mmul(weightRows);
    }

    @Override
    public INDArray activate(final boolean training) {
        final INDArray rows = preOutput(training);
        final INDArray ret = layerConf().getActivationFn().getActivation(rows, training);

        if (maskArray != null) {
            applyMask(ret);
        }

        return ret;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    protected void applyDropOutIfNecessary(final boolean training) {
        // no-op
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(final INDArray epsilon) {
        if (maskArray != null) {
            epsilon.muliColumnVector(maskArray);
        }

        final INDArray dLdw = denseInput.transpose().mmul(epsilon); // Shape: [nIn,minibatch]x[minibatch,nOut] = [nInEffective, nOut]

        // The gradient matrix is reused to avoid re-creating it over and over.
        // That's why we assign all values to zero between runs.

        final INDArray weightGradients = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);
        weightGradients.assign(0);

        final ScatterUpdate op = new ScatterUpdate(weightGradients, dLdw, intIndices, SCATTER_DIM, ScatterUpdate.UpdateOp.ADD);
        Nd4j.getExecutioner().exec(op);

        final Gradient ret = new DefaultGradient();
        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGradients);

        return new Pair<>(ret, null); // Don't bother returning epsilons: no layer below this one...
    }

    @Override
    public boolean hasBias() {
        return false;
    }

    public INDArray getDenseInput(final int[] intIndices) {
        final int miniBatchSize = input.size(0); // Rows
        final INDArray denseInput = Nd4j.create(miniBatchSize, intIndices.length);

        final CustomOp op = DynamicCustomOp.builder("firas_sparse")
                .addInputs(input)
                .addIntegerArguments(intIndices)
                .addOutputs(denseInput)
                .build();

        Nd4j.getExecutioner().exec(op);

        return denseInput;
    }

    public int[] getIntIndices() {
        final int columns = input.size(1);
        final INDArray indexes = input.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 2, columns));

        // Filter out all -1 values (which are padding)
        return Arrays.stream(indexes.dup().data().asInt()).filter(x -> x != -1).distinct().toArray();
    }
}
