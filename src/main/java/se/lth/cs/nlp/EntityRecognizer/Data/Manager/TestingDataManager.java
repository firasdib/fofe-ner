package se.lth.cs.nlp.EntityRecognizer.Data.Manager;

import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import se.lth.cs.nlp.EntityRecognizer.Data.EntitySentence;
import se.lth.cs.nlp.EntityRecognizer.Data.Iterator.DataIterator;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.MetaData;

import java.util.*;

public class TestingDataManager extends DataManager {
    public TestingDataManager(final MetaData metaData) {
        super(metaData);
    }

    private List<EntitySentence> getEntitySentences() {
        final List<EntitySentence> entitySentences = new ArrayList<>();

        // Add all n-grams to the entity sentence map
        metaData.getCorpus().getSentences().forEach(sentence ->
                sentence.getAllNGrams().forEach(nGram ->
                        entitySentences.add(new EntitySentence(sentence, nGram))));

        return entitySentences;
    }

    @NotNull
    @Override
    protected MultiDataSetIterator create() {
        return new DataIterator(this.getEntitySentences(), featureSet, metaData.getEntityTypes());
    }
}
