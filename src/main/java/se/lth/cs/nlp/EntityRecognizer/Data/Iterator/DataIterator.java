package se.lth.cs.nlp.EntityRecognizer.Data.Iterator;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.Corpus.NGram;
import se.lth.cs.nlp.EntityRecognizer.Data.EntitySentence;
import se.lth.cs.nlp.EntityRecognizer.Data.NGramDataSet;
import se.lth.cs.nlp.EntityRecognizer.Features.FeatureSet;

import java.util.*;

public class DataIterator implements MultiDataSetIterator {
    private final List<EntitySentence> sentences;
    private final FeatureSet featureSet;
    private final List<String> entityTypes;
    private final int batchSize;
    private int sentenceIdx = 0;

    public DataIterator(final List<EntitySentence> sentences, final FeatureSet featureSet,
                        final List<String> entityTypes) {
        this.sentences = sentences;
        this.featureSet = featureSet;
        this.entityTypes = entityTypes;

        batchSize = Config.getInt("batchSize");
    }

    @Override
    public boolean hasNext() {
        return sentences.size() > sentenceIdx;
    }

    public MultiDataSet getData(final int size) {
        final int amountOfFeatures = featureSet.getFeatures().size();

        final List<List<INDArray>> features = new ArrayList<>(amountOfFeatures);
        final List<INDArray> outputs = new ArrayList<>();
        final List<NGram> nGrams = new ArrayList<>();

        for (int i = 0; i < amountOfFeatures; i++) {
            features.add(new ArrayList<>(size));
        }

        int featureCount = 0;

        // wordInputs and charInputs will be same length
        while (hasNext() && featureCount++ < size) {
            final EntitySentence entitySentence = sentences.get(sentenceIdx++);

            final INDArray[] feature = featureSet.getFeatureVector(entitySentence);

            for (int i = 0; i < feature.length; i++) {
                features.get(i).add(feature[i]);
            }

            final NGram nGram = entitySentence.getNGram();

            // Inference => null
            if (nGram.getEntityType() != null) {
                // In some cases we don't have any output vector, so we just insert a zero vector
                final INDArray outputVector = Nd4j.create(1, entityTypes.size());

                final int entityIdx = entityTypes.indexOf(nGram.getEntityType());

                if (entityIdx == -1) {
                    throw new RuntimeException("Unknown entity type " + nGram.getEntityType());
                }

                outputVector.put(0, entityIdx, 1);

                outputs.add(outputVector);
            }

            nGrams.add(nGram);
        }

        return new NGramDataSet(features, outputs, nGrams);
    }

    public int getBatchCount() {
        return (int) Math.max(1, Math.ceil(sentences.size() / batchSize));
    }

    @Override
    public MultiDataSet next() {
        return getData(batchSize);
    }

    @Override
    public MultiDataSet next(final int num) {
        return getData(num);
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {

    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
//        sentenceIdx = 0;
    }

//    public void reset(final List<EntitySentence> entitySentences) {
//        this.sentences = entitySentences;
//
//        reset();
//    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
    }
}
