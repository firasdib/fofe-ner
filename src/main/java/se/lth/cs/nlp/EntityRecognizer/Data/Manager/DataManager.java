package se.lth.cs.nlp.EntityRecognizer.Data.Manager;

import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import se.lth.cs.nlp.EntityRecognizer.Features.FeatureSet;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.MetaData;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.NNFeatures;

public abstract class DataManager implements Iterable<MultiDataSet> {
    protected final FeatureSet featureSet;
    protected final MetaData metaData;

    public DataManager(final MetaData metaData) {
        this.metaData = metaData;

        this.featureSet = NNFeatures.create(metaData.getW2v());
    }

    @NotNull
    public MultiDataSetIterator iterator() {
        return create();
    }

    @NotNull
    protected abstract MultiDataSetIterator create();
}
