package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.StackVertex;
import org.deeplearning4j.nn.conf.graph.UnstackVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.solvers.accumulation.EncodedGradientsAccumulator;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Corpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.EntityType;
import se.lth.cs.nlp.EntityRecognizer.Data.Manager.DataManager;
import se.lth.cs.nlp.EntityRecognizer.Data.Manager.TrainingDataManager;
import se.lth.cs.nlp.EntityRecognizer.FOFE.CharContext;
import se.lth.cs.nlp.EntityRecognizer.Features.Feature;
import se.lth.cs.nlp.EntityRecognizer.Features.FeatureSet;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Evaluator.CoNLLEvaluator;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Evaluator.DefaultEvaluator;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Evaluator.DocforiaEvaluator;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Evaluator.Evaluator;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.ProjectionLayer.ProjectionLayer;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.ProjectionLayer.ProjectionType;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Schedules.DropoutSchedule;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Schedules.LearningRateSchedule;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Updater.Momentum;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class NeuralNetwork {
    private static final Logger log = LoggerFactory.getLogger(NeuralNetwork.class);

    private final int epochCount;
    private final String modelSaveFolder;
    private final ParallelWrapper parallelWrapper;
    private final ComputationGraph model;
    private final String modelName;
    private final NetworkMode networkMode;
    private final Corpus devEvaluationCorpus;
    private final Corpus evaluationCorpus;
    private final Word2Vec w2v;

    public NeuralNetwork(final Builder builder) {
        parallelWrapper = builder.parallelWrapper;
        modelSaveFolder = builder.modelSaveFolder;
        modelName = builder.modelName;
        model = builder.model;
        networkMode = builder.networkMode;
        devEvaluationCorpus = builder.devEvaluationCorpus;
        evaluationCorpus = builder.evaluationCorpus;
        w2v = builder.w2v;
        epochCount = builder.epochCount;
    }

    public synchronized void evaluate(final Corpus corpus) {
        final MetaData metaData = new MetaData(corpus, w2v, networkMode);
        final Evaluator evaluator;

        switch (networkMode) {
            case CONLL2002:
            case CONLL2003:
                evaluator = new CoNLLEvaluator();
                break;
            case DOCFORIA_SUC:
            case DOCFORIA_CONLL:
            case DOCFORIA_TAC:
                evaluator = new DocforiaEvaluator();
                break;
            case TAC:
            case SUC:
            default:
                evaluator = new DefaultEvaluator();
        }

        evaluator.evaluate(model, metaData);
    }

    public synchronized void train(final Corpus corpus) {
        final long startTime = System.currentTimeMillis();
        long avg = 0;

        log.info("Training model...");

        final MetaData metaData = new MetaData(corpus, w2v, networkMode);

        final DataManager trainingDataManager = new TrainingDataManager(metaData);

        for (int i = 0; i < epochCount; i++) {
            final int epoch = i + 1;

            log.info("Epoch {}", epoch);

            final long epochStartTime = System.currentTimeMillis();

            final MultiDataSetIterator trainingIterator = trainingDataManager.iterator();

            if (parallelWrapper != null) {
                parallelWrapper.fit(trainingIterator);
            } else {
                model.fit(trainingIterator);
            }

            final long epochTotalTime = System.currentTimeMillis() - epochStartTime;

            avg += epochTotalTime;

            if (epoch < epochCount) {
                final long avgTime = avg / epoch;
                log.info("Epoch {} took {} to finish (avg. {}); {}% training done.",
                        epoch,
                        NNUtils.formatTime(epochTotalTime),
                        NNUtils.formatTime(avgTime),
                        Math.round(epoch * 100.0 / (double) epochCount));
                log.info("Estimated {} left until finish.\n", NNUtils.formatTime(avgTime * (epochCount - epoch)));
            }

            if (epoch % 10 == 0) {
                if (devEvaluationCorpus != null) {
                    log.info("Evaluating dev corpus...");
                    evaluate(devEvaluationCorpus);
                }

                if (evaluationCorpus != null) {
                    log.info("Evaluating test corpus...");
                    evaluate(evaluationCorpus);
                }
            }

            if (modelName != null && epoch % 10 == 0) {
                saveModel(modelName + "_epoch_" + epoch);
            }
        }

        if (modelName != null) {
            saveModel(modelName);
        }

        log.info("Training done! Took {}\n", NNUtils.formatTime(System.currentTimeMillis() - startTime));

        if (parallelWrapper != null) {
            parallelWrapper.shutdown();
        }
    }

    //
    // Helpers

    public void saveModel(final String modelName) {
        if (modelName == null) {
            return;
        }

        final File folder = new File(modelSaveFolder);

        if (!folder.exists() && !folder.mkdirs()) {
            throw new IllegalArgumentException("The path '" + folder.getAbsolutePath() + "' is a directory.");
        }

        final File locationToSave = new File(folder, modelName + ".zip");

        if (locationToSave.exists() && !locationToSave.delete()) {
            log.error("Unable to delete model '{}' before writing.", folder.getAbsolutePath());

            return;
        }

        try {
            ModelSerializer.writeModel(model, locationToSave, true);

            log.info("Successfully saved model '{}'", locationToSave.getName());
        } catch (final IOException e) {
            log.error("Unable to save neural network model", e);
        }
    }

    public NetworkMode getNetworkMode() {
        return networkMode;
    }

    //
    // Builder

    public static class Builder {
        private final String modelSaveFolder = Config.getString("modelFolder");
        private final String uiServerPort = Config.getString("uiServerPort");
        private final int printFrequency = Config.getInt("printFrequency");
        private final int epochCount = Config.getInt("epochCount");

        // Projections
        private final int dimWordProjection = Config.getInt("dimWordProjection");
        private final int dimCharProjection = Config.getInt("dimCharProjection");

        private final int kernelSizeStart = Config.getInt("kernelSizeStart");
        private final int kernelSizeEnd = Config.getInt("kernelSizeEnd");

        // NN modelConfig
        private final int rngSeed = 1337;
        private final int neuronCount = Config.getInt("neuronCount");
        private final double learningRate = Config.getDouble("learningRate");
        private final double l2Regularization = Config.getDouble("l2reg");
        private final double dropout = Config.getDouble("dropout");
        private final double adamVarDecay = Config.getDouble("adamVarDecay");
        private final double adamMeanDecay = Config.getDouble("adamMeanDecay");
        private final double adamEpsilon = Config.getDouble("adamEpsilon");
        private final double momentum = Config.getDouble("momentum");
        private final String optimizer = Config.getString("optimizer");

        // Parallel wrapper
        private final boolean useParallelWrapper = Config.getBool("useParallelWrapper");
        private final int numberOfWorkers = Config.getInt("numberOfWorkers");

        // Feature selection
        private final boolean useCharFOFEFeatures = Config.getBool("useCharFOFEFeatures");
        private final boolean useCharCNNFeatures = Config.getBool("useCharCNNFeatures");
        private final boolean useWordFOFEFeaturesCIS = Config.getBool("useWordFOFEFeaturesCIS");
        private final boolean useWordFOFEFeaturesCS = Config.getBool("useWordFOFEFeaturesCS");
        private final boolean useInitialCharFOFEFeatures = Config.getBool("useInitialCharFOFEFeatures");
        private final boolean chineseMode = Config.getBool("chineseMode");

        private String modelName = null;
        private String loadModelPath = null;
        private boolean useStatsUI = false;
        private NetworkMode networkMode = null;
        private String embeddingsCS = null;
        private String embeddingsCIS = null;
        private Word2Vec w2v = null;
        private Corpus devEvaluationCorpus = null;
        private Corpus evaluationCorpus = null;
        private ComputationGraph model = null;
        private ParallelWrapper parallelWrapper = null;

        public Builder loadModel(final String modelName) {
            loadModelPath = modelName;

            return this;
        }

        public Builder setMode(final NetworkMode networkMode) {
            this.networkMode = networkMode;

            return this;
        }

        public Builder useStatsUI(final boolean useStatsUI) {
            this.useStatsUI = useStatsUI;

            return this;
        }

        public Builder setEmbeddingsCS(final String embeddingsCS) {
            this.embeddingsCS = embeddingsCS;

            return this;
        }

        public Builder setEmbeddingsCIS(final String embeddingsCIS) {
            this.embeddingsCIS = embeddingsCIS;

            return this;
        }

        public Builder setModelName(final String modelName) {
            this.modelName = modelName;

            return this;
        }

        public Builder devEvaluationCorpus(final Corpus devEvaluationCorpus) {
            this.devEvaluationCorpus = devEvaluationCorpus;

            return this;
        }

        public Builder evaluationCorpus(final Corpus evaluationCorpus) {
            this.evaluationCorpus = evaluationCorpus;

            return this;
        }

        public NeuralNetwork build() {
            if (networkMode == null) {
                throw new IllegalStateException("You must provide a network mode.");
            }

            if (embeddingsCS == null || embeddingsCIS == null) {
                throw new IllegalStateException("You must provide both case-sensitive and case-insensitive word embeddings.");
            }


            if (loadModelPath == null) {
                w2v = new Word2Vec(embeddingsCS, embeddingsCIS);

                if (modelName == null || modelName.equals("")) {
                    log.info("Data will NOT be saved for this run!");
                } else {
                    log.info("Model will be saved to '{}'", modelSaveFolder + modelName + ".zip");
                }

                log.info("Building model...");
                model = new ComputationGraph(createConfig(w2v));
            } else {
                w2v = new Word2Vec(embeddingsCS, embeddingsCIS, true);

                log.info("Loading model from '{}'...", loadModelPath);

                try {
                    model = ModelSerializer.restoreComputationGraph(loadModelPath);
                } catch (final IOException e) {
                    log.error("Unable to load neural network model", e);

                    System.exit(1);

                    return null;
                }
            }

            model.init();

            final List<IterationListener> listeners = new ArrayList<>();
            listeners.add(new ScoreIterationListener(printFrequency));
            listeners.add(new PerformanceListener(printFrequency));

            if (useStatsUI) {
                log.info("Using port {}", uiServerPort);
                System.setProperty("org.deeplearning4j.ui.port", uiServerPort);
                System.setProperty("play.crypto.secret", "firas");

                log.info("Enabling stats UI...");
                final UIServer uiServer = UIServer.getInstance();
                final StatsStorage statsStorage;

                if (modelName == null) {
                    statsStorage = new InMemoryStatsStorage();
                } else {
                    statsStorage = new FileStatsStorage(new File(modelSaveFolder, modelName + ".stats"));
                }

                uiServer.attach(statsStorage);

                listeners.add(new StatsListener(statsStorage, 100));
            }

            model.setListeners(listeners);

            if (useParallelWrapper) {
                parallelWrapper = new ParallelWrapper.Builder<>(model)
                        .prefetchBuffer(numberOfWorkers)
                        .workers(numberOfWorkers)
                        .trainingMode(ParallelWrapper.TrainingMode.CUSTOM)
                        .gradientsAccumulator(new EncodedGradientsAccumulator.Builder(numberOfWorkers)
                                .encodingThreshold(1E-5)
                                .memoryParameters(250 * 1024 * 1024L, 10)
                                .build())
                        .workspaceMode(WorkspaceMode.SINGLE)
                        .build();
            }

            log.info("Model construction done!");

            return new NeuralNetwork(this);
        }

        protected ComputationGraphConfiguration createConfig(final Word2Vec w2v) {
            final FeatureSet probe = NNFeatures.create(w2v);
            final int outputSize = EntityType.getEntityTypes(networkMode).size();

            log.info("=================================================");
            log.info("The following features are being used:");
            for (final Feature feature : probe.getFeatures()) {
                log.info("[{}]\t{}", feature.getFeatureType(), feature.getName());
            }
            log.info("=================================================");

            //
            // Define the basic settings

            final ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                    .miniBatch(true)
                    .trainingWorkspaceMode(WorkspaceMode.SINGLE)
                    .inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .weightInit(WeightInit.RELU_UNIFORM)
                    .updater(this.createOptimizer(optimizer))
                    .activation(Activation.LEAKYRELU)
                    .seed(rngSeed)
                    .convolutionMode(ConvolutionMode.Truncate)
                    .graphBuilder();

            //
            // Set the inputs to the system

            graphBuilder.addInputs(probe.getFeatures().stream()
                    .map(Feature::getName)
                    .collect(Collectors.toList()));

            final List<String> hiddenLayerInputNames = new ArrayList<>();

            // Create the mapping between feature (input name) and stack projection layer
            final Map<ProjectionType, List<String>> feature2layer = new HashMap<>();

            // Piggyback and find the char cnn input, if any
            String charCnnInputName = null;

            for (final Feature feature : probe.getFeatures()) {
                final ProjectionType projectionType;

                switch (feature.getFeatureType()) {
                    case WORD:
                        projectionType = feature.isCaseSensitive() ? ProjectionType.WORD : ProjectionType.WORD_LC;
                        break;
                    case CHAR:
                        projectionType = ProjectionType.CHAR;
                        break;
                    case CHAR_CNN:
                        charCnnInputName = feature.getName();
                        continue;
                    default:
                        continue;
                }

                feature2layer.putIfAbsent(projectionType, new ArrayList<>());
                feature2layer.get(projectionType).add(feature.getName());
            }

            //
            // Word projection

            if (useWordFOFEFeaturesCS) {
                final String stackWord = getProjectionStackName(ProjectionType.WORD);

                graphBuilder
                        .addVertex(stackWord, new StackVertex(),
                                feature2layer.get(ProjectionType.WORD).toArray(new String[0]))
                        .addLayer(getProjectionName(ProjectionType.WORD),
                                new ProjectionLayer.Builder()
                                        .setW2VWeight(w2v.getWeights(true))
                                        .updater(this.createOptimizer("sgd", learningRate / 2))
                                        .build(), stackWord);
            }

            //
            // Word lower case projection

            if (useWordFOFEFeaturesCIS) {
                final String stackWordLC = getProjectionStackName(ProjectionType.WORD_LC);

                graphBuilder
                        .addVertex(stackWordLC, new StackVertex(),
                                feature2layer.get(ProjectionType.WORD_LC).toArray(new String[0]))
                        .addLayer(getProjectionName(ProjectionType.WORD_LC),
                                new ProjectionLayer.Builder()
                                        .setW2VWeight(w2v.getWeights(false))
                                        .updater(this.createOptimizer("sgd", learningRate / 2))
                                        .build(), stackWordLC);
            }

            //
            // Char FOFE projection

            if (!chineseMode && (useCharFOFEFeatures || useInitialCharFOFEFeatures)) {
                final String stackChar = getProjectionStackName(ProjectionType.CHAR);

                graphBuilder
                        .addVertex(stackChar, new StackVertex(),
                                feature2layer.get(ProjectionType.CHAR).toArray(new String[0]))
                        .addLayer(getProjectionName(ProjectionType.CHAR),
                                new ProjectionLayer.Builder()
                                        .nIn(CharContext.CHAR_DIMENSIONS)
                                        .nOut(dimCharProjection)
                                        .updater(this.createOptimizer("sgd", learningRate / 2))
                                        .build(), stackChar);
            }

            //
            // Add convolution

            if (!chineseMode && useCharCNNFeatures) {
                final List<String> cnnOutput = new ArrayList<>(kernelSizeEnd - kernelSizeStart);

                for (int i = kernelSizeStart; i <= kernelSizeEnd; i++) {
                    final String name = "char_cnn_" + i;

                    cnnOutput.add(name);

                    graphBuilder
                            .addLayer(name, new ConvolutionLayer.Builder()
                                    .nIn(1)
                                    .nOut(getCNNOut())
                                    .kernelSize(2, dimCharProjection)
                                    .stride(1, dimCharProjection)
                                    .padding(0, 0)
                                    .updater(this.createOptimizer(optimizer))
                                    .build(), charCnnInputName);
                }

                graphBuilder
                        .addLayer("cnn_pool", new GlobalPoolingLayer.Builder()
                                .poolingType(PoolingType.MAX)
                                .build(), cnnOutput.toArray(new String[0]))
                        .addVertex("merge_cnn", new MergeVertex(), "cnn_pool");

                hiddenLayerInputNames.add("merge_cnn");
            }

            // Add the unstack vertices for the stacked input earlier.
            // This does not apply to CHAR_CNN features since they have their
            // own projection layer which they don't share.

            for (final ProjectionType pt : feature2layer.keySet()) {
                final List<String> inputNames = feature2layer.get(pt);
                final String projectionName = getProjectionName(pt);

                int i = 0;
                for (final String inputName : inputNames) {
                    final String vertexName = "unstack_" + i + "_" + inputName;

                    graphBuilder.addVertex(vertexName, new UnstackVertex(i, inputNames.size()), projectionName);

                    hiddenLayerInputNames.add(vertexName);

                    i++;
                }
            }

            // Add merge vertex which merges all projections into one
            graphBuilder.addVertex("mergeFeatures", new MergeVertex(),
                    hiddenLayerInputNames.toArray(new String[0]));

            //
            // Add the actual network that does predictions

            final int hiddenInputSize = featureLength(probe.getFeatures());

            log.info("Hidden layer input size = {}", hiddenInputSize);

            graphBuilder
                    .addLayer("hidden1", new DenseLayer.Builder()
                            .nIn(hiddenInputSize)
                            .nOut(neuronCount)
                            .dropOut(new Dropout(new DropoutSchedule(dropout, epochCount)))
                            .l2(l2Regularization)
                            .activation(Activation.IDENTITY)
                            .build(), "mergeFeatures")
                    .addLayer("batch_norm_1", new BatchNormalization.Builder()
                            .nIn(neuronCount)
                            .nOut(neuronCount)
                            .build(), "hidden1")
                    .addLayer("activation1", new ActivationLayer.Builder()
                            .activation(Activation.LEAKYRELU)
                            .build(), "batch_norm_1")
                    .addLayer("hidden2", new DenseLayer.Builder()
                            .nIn(neuronCount)
                            .nOut(neuronCount)
                            .dropOut(new Dropout(new DropoutSchedule(dropout, epochCount)))
                            .l2(l2Regularization)
                            .updater(this.createOptimizer(optimizer))
                            .activation(Activation.IDENTITY)
                            .build(), "activation1")
                    .addLayer("batch_norm_2", new BatchNormalization.Builder()
                            .nIn(neuronCount)
                            .nOut(neuronCount)
                            .build(), "hidden2")
                    .addLayer("activation2", new ActivationLayer.Builder()
                            .activation(Activation.LEAKYRELU)
                            .build(), "batch_norm_2")
                    .addLayer("hidden3", new DenseLayer.Builder()
                            .nIn(neuronCount)
                            .nOut(neuronCount)
                            .dropOut(new Dropout(new DropoutSchedule(dropout, epochCount)))
                            .l2(l2Regularization)
                            .build(), "activation2")
                    .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                            .nIn(neuronCount)
                            .nOut(outputSize)
                            .activation(Activation.SOFTMAX)
                            .build(), "hidden3")
                    .setOutputs("output")
                    .pretrain(false)
                    .backprop(true);

            return graphBuilder.build();
        }

        //
        // Private help methods to keep network generation simple and consistent

        private int featureLength(final List<Feature> features) {
            int len = 0;

            for (final Feature feature : features) {
                len += featureLength(feature);
            }

            return len;
        }

        private int featureLength(final Feature feature) {
            switch (feature.getFeatureType()) {
                case WORD:
                    return dimWordProjection;
                case CHAR:
                    return dimCharProjection;
                case CHAR_CNN:
                    // Sum of all kernel depths
                    return getCNNOut() * (kernelSizeEnd - kernelSizeStart + 1);
                default:
                    return 0;
            }
        }

        private String getProjectionStackName(final ProjectionType pt) {
            switch (pt) {
                case WORD:
                    return "stackWordProjection";
                case WORD_LC:
                    return "stackWordProjectionLC";
                case CHAR:
                    return "stackCharProjection";
                default:
                    return null;
            }
        }

        private String getProjectionName(final ProjectionType pt) {
            switch (pt) {
                case WORD:
                    return "wordProjection";
                case WORD_LC:
                    return "wordProjectionLC";
                case CHAR:
                    return "charProjection";
                case CHAR_CNN:
                    return "charCNNProjection";
                default:
                    throw new RuntimeException("No such projection type '" + pt + "'");
            }
        }

        private IUpdater createOptimizer(final String optimizer, final double learningRate) {
            final LearningRateSchedule lrSchedule = new LearningRateSchedule(learningRate, epochCount);

            switch (optimizer.toLowerCase()) {
                case "nadam":
                    return Nadam.builder()
                            .beta1(adamMeanDecay)
                            .beta2(adamVarDecay)
                            .epsilon(adamEpsilon)
                            .learningRateSchedule(lrSchedule)
                            .build();
                case "adam":
                    return Adam.builder()
                            .beta1(adamMeanDecay)
                            .beta2(adamVarDecay)
                            .epsilon(adamEpsilon)
                            .learningRateSchedule(lrSchedule)
                            .build();
                case "momentum":
                    return new Momentum(lrSchedule, momentum);
                case "nesterovs":
                    return new Nesterovs(lrSchedule, momentum);
                case "sgd":
                    return new Sgd(lrSchedule);
                default:
                    throw new RuntimeException("Unknown optimizer '" + optimizer + "'");
            }
        }

        private IUpdater createOptimizer(final String optimizer) {
            return createOptimizer(optimizer, learningRate);
        }

        private int getCNNOut() {
            // +1 since we want to include each kernel defined
            return CharContext.CHAR_DIMENSIONS / (kernelSizeEnd - kernelSizeStart + 1);
        }
    }
}
