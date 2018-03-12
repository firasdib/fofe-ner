package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.ProjectionLayer;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

public class ProjectionLayer extends FeedForwardLayer {

    private final INDArray w2vWeights;

    @SuppressWarnings("unused")
    public ProjectionLayer() {
        // We need a no-arg constructor so we can deserialize the configuration from JSON or YAML format
        // Without this, you will likely get an exception like the following:
        // com.fasterxml.jackson.databind.JsonMappingException: No suitable constructor found for type [simple type, class org.deeplearning4j.examples.misc.customlayers.layer.CustomLayer]: can not instantiate from JSON object (missing default constructor or creator, or perhaps need to add/enable type information?)

        w2vWeights = null;
    }

    private ProjectionLayer(final Builder builder) {
        super(builder);

        this.w2vWeights = builder.w2vWeights;
    }

    @Override
    public Layer instantiate(final NeuralNetConfiguration conf, final Collection<IterationListener> iterationListeners,
                             final int layerIndex, final INDArray layerParamsView, final boolean initializeParams) {
        final ProjectionLayerImpl projectionLayer = new ProjectionLayerImpl(conf);
        projectionLayer.setListeners(iterationListeners);
        projectionLayer.setIndex(layerIndex);

        projectionLayer.setParamsViewArray(layerParamsView);

        final Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        projectionLayer.setParamTable(paramTable);
        projectionLayer.setConf(conf);

        if (w2vWeights != null) {
            // Optionally initialize layer weights with word2vec model weights
            projectionLayer.getParam(DefaultParamInitializer.WEIGHT_KEY).assign(w2vWeights);
        }

        return projectionLayer;
    }

    @Override
    public ParamInitializer initializer() {
        return DefaultParamInitializer.getInstance();
    }

    @Override
    public LayerMemoryReport getMemoryReport(final InputType inputType) {
        // Basically a dense layer, but no dropout is possible here, and no epsilons
        final InputType outputType = getOutputType(-1, inputType);

        final int actElementsPerEx = outputType.arrayElementsPerExample();
        final int numParams = initializer().numParams(this);
        final int updaterStateSize = (int) getIUpdater().stateSize(numParams);

        // Embedding layer does not use caching.
        // Inference: no working memory - just activations (pullRows)
        // Training: preout op, the only in-place ops on epsilon (from layer above) + assign ops

        return new LayerMemoryReport.Builder(layerName, EmbeddingLayer.class, inputType, outputType)
                .standardMemory(numParams, updaterStateSize)
                .workingMemory(0, 0, 0, actElementsPerEx)
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) // No caching
                .build();
    }

    public static class Builder extends FeedForwardLayer.Builder<Builder> {
        private INDArray w2vWeights = null;

        public Builder setW2VWeight(final INDArray weights) {
            if (weights != null) {
                w2vWeights = weights;

                // These "dummy" values are necessary to get the right weight matrix initialized
                nIn = w2vWeights.rows();
                nOut = w2vWeights.columns();
            }

            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public ProjectionLayer build() {
            return new ProjectionLayer(this);
        }
    }
}
