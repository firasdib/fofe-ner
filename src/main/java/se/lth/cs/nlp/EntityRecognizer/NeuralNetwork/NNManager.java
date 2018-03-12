package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork;

import se.lth.cs.nlp.EntityRecognizer.CLI.EvalCommand;
import se.lth.cs.nlp.EntityRecognizer.CLI.TrainCommand;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Corpus;

public class NNManager {
    public static void create(final TrainCommand tc) {
        final String embeddingsCS = Config.getString("word2vecModelCS");
        final String embeddingsCIS = Config.getString("word2vecModelCIS");

        final Corpus trainingCorpus = Corpus.createCorpus(tc.networkMode, tc.trainingFile);
        final Corpus devEvaluationCorpus;
        final Corpus evaluationCorpus;

        if (tc.devEvaluationFile != null) {
            devEvaluationCorpus = Corpus.createCorpus(tc.networkMode, tc.devEvaluationFile);
        } else {
            devEvaluationCorpus = null;
        }

        if (tc.evaluationFile != null) {
            evaluationCorpus = Corpus.createCorpus(tc.networkMode, tc.evaluationFile);
        } else {
            evaluationCorpus = null;
        }

        final NeuralNetwork neuralNetwork = new NeuralNetwork.Builder()
                .setModelName(tc.saveModelName)
                .useStatsUI(tc.useStatsUI)
                .setMode(tc.networkMode)
                .setEmbeddingsCS(embeddingsCS)
                .setEmbeddingsCIS(embeddingsCIS)
                .devEvaluationCorpus(devEvaluationCorpus)
                .evaluationCorpus(evaluationCorpus)
                .build();

        neuralNetwork.train(trainingCorpus);

        if (tc.evaluationFile != null) {
            neuralNetwork.evaluate(evaluationCorpus);
        }
    }

    public static void load(final EvalCommand ec) {
        final String embeddingsCS = Config.getString("word2vecModelCS");
        final String embeddingsCIS = Config.getString("word2vecModelCIS");

        final NeuralNetwork neuralNetwork = new NeuralNetwork.Builder()
                .loadModel(ec.loadModelName)
                .setMode(ec.networkMode)
                .setEmbeddingsCS(embeddingsCS)
                .setEmbeddingsCIS(embeddingsCIS)
                .build();

        final Corpus evaluationCorpus = Corpus.createCorpus(ec.networkMode, ec.evaluationFile);

        neuralNetwork.evaluate(evaluationCorpus);
    }

    public static NeuralNetwork load(final String modelName, final NetworkMode networkMode) {
        final String embeddingsCS = Config.getString("word2vecModelCS");
        final String embeddingsCIS = Config.getString("word2vecModelCIS");

        return new NeuralNetwork.Builder()
                .loadModel(modelName)
                .setEmbeddingsCS(embeddingsCS)
                .setEmbeddingsCIS(embeddingsCIS)
                .setMode(networkMode)
                .build();
    }
}
