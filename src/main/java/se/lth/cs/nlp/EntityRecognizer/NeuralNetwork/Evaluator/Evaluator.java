package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Evaluator;

import com.google.common.collect.Range;
import com.google.common.collect.RangeSet;
import com.google.common.collect.TreeRangeSet;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.Corpus.EntityType;
import se.lth.cs.nlp.EntityRecognizer.Corpus.NGram;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Sentence;
import se.lth.cs.nlp.EntityRecognizer.Data.Iterator.DataIterator;
import se.lth.cs.nlp.EntityRecognizer.Data.Manager.DataManager;
import se.lth.cs.nlp.EntityRecognizer.Data.Manager.TestingDataManager;
import se.lth.cs.nlp.EntityRecognizer.Data.NGramDataSet;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.MetaData;

import java.util.*;

public abstract class Evaluator {
    protected final String algoLongest;
    protected final String algoHighest;
    protected final String pruningAlgorithm;
    protected final double ngramThreshold;
    protected MetaData metaData;
    protected int batchCount;

    public Evaluator() {
        ngramThreshold = Config.getDouble("nGramThreshold");
        pruningAlgorithm = Config.getString("pruningAlgorithm");
        algoHighest = "highest";
        algoLongest = "longest";
    }

    protected abstract void eval(final INDArray modelOutput, final NGramDataSet dataSet);

    protected abstract void preEval();

    protected abstract void postEval(final long startTime);

    public void evaluate(final ComputationGraph model, final MetaData metaData) {
        if (!pruningAlgorithm.equalsIgnoreCase(algoHighest) &&
                !pruningAlgorithm.equalsIgnoreCase(algoLongest)) {
            throw new RuntimeException("Invalid pruning algorithm chosen: must be either '" + algoHighest
                    + "' or '" + algoLongest + "'");
        }

        final long startTime = System.currentTimeMillis();

        this.metaData = metaData;

        final DataManager testingDataManager = new TestingDataManager(metaData);
        final MultiDataSetIterator iterator = testingDataManager.iterator();
        batchCount = ((DataIterator) iterator).getBatchCount();

        preEval();

        while (iterator.hasNext()) {
            final NGramDataSet dataSet = (NGramDataSet) iterator.next();
            final INDArray modelOutput = model.outputSingle(false, dataSet.getFeatures());

            eval(modelOutput, dataSet);

            setEntityTags(modelOutput, dataSet);
        }

        pruneEntityTags();

        postEval(startTime);
    }

    /**
     * Trickle down the predicted entity tags of the n-gram onto the words.
     * Longest match wins.
     */
    private void pruneEntityTags() {
        for (final Sentence sentence : metaData.getCorpus().getSentences()) {
            final RangeSet<Integer> ranges = TreeRangeSet.create();
            final List<NGram> prunedNGrams = new ArrayList<>();

            final List<NGram> allSentenceNGrams = new ArrayList<>(sentence.getAllNGrams());

            // Ascending sort
            if (pruningAlgorithm.equalsIgnoreCase(algoHighest)) {
                allSentenceNGrams.sort((n1, n2) -> {
                    if (n1.getPredictedProbability() > n2.getPredictedProbability()) {
                        return 1;
                    }

                    if (n1.getPredictedProbability() < n2.getPredictedProbability()) {
                        return -1;
                    }

                    // noinspection UseCompareMethod
                    if (n1.length() > n2.length()) {
                        return 1;
                    }

                    if (n1.length() < n2.length()) {
                        return -1;
                    }

                    return 0;
                });
            } else {
                allSentenceNGrams.sort((n1, n2) -> {
                    // If its longer it goes up
                    if (n1.length() > n2.length()) {
                        return 1;
                    }

                    // If its shorter it goes down
                    if (n1.length() < n2.length()) {
                        return -1;
                    }

                    // If its the same length but less probable, push it down
                    // noinspection UseCompareMethod
                    if (n1.getPredictedProbability() < n2.getPredictedProbability()) {
                        return -1;
                    }

                    // If its the same length but more probable. push it up
                    if (n1.getPredictedProbability() > n2.getPredictedProbability()) {
                        return 1;
                    }

                    return 0;
                });
            }

            // Its sorted ascending, reverse it to get descending
            Collections.reverse(allSentenceNGrams);

            for (final NGram nGram : allSentenceNGrams) {
                final Range<Integer> range = Range.closedOpen(nGram.getStartIndex(), nGram.getEndIndex());

                // Check that there is no intersection with another range
                if (!nGram.getPredictedEntityTag().equals(EntityType.NONE) &&
                        nGram.getPredictedProbability() > ngramThreshold && ranges.subRangeSet(range).isEmpty()) {
                    prunedNGrams.add(nGram);

                    ranges.add(range);
                } else {
                    // This nGram is intersecting with another n-gram of higher probability.
                    // We set all of these words to NONE, and in the code below, it might be set
                    // to another value.

                    nGram.getWords().forEach(word -> word.setPredictedEntityTag(EntityType.NONE));
                }

            }

            // Go through the pruned and validated ranges
            for (final NGram nGram : prunedNGrams) {
                nGram.setSurvivedPruning(true);

                nGram.getWords().forEach(word -> word.setPredictedEntityTag(nGram.getPredictedEntityTag()));
            }
        }
    }

    /**
     * This method just takes the evaluated output and extends it to the n-grams.
     * The n-grams will then have a predicted entity type fixed to them.
     * Note that the underlying words that make up this n-gram are not annotated yet!
     * That is done in the final pruning after all n-grams have been classified.
     *
     * @param modelOutput The model output
     * @param dataSet     The data set that was fed to the network to produce `modelOutput`
     */
    private void setEntityTags(final INDArray modelOutput, final NGramDataSet dataSet) {
        for (int i = 0; i < modelOutput.rows(); i++) {
            final INDArray nGramOutput = modelOutput.getRow(i);
            final NGram nGram = dataSet.getNGrams().get(i);

            // Find highest probability for this n-gram
            final int idx = Nd4j.argMax(nGramOutput).getInt(0);
            final double maxProb = nGramOutput.getDouble(0, idx);
            final String entityType = metaData.getEntityTypes().get(idx);

            nGram.setPredictedEntityTag(maxProb, entityType);
        }
    }
}
