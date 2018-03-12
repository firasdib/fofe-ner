package se.lth.cs.nlp.EntityRecognizer.Data;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import se.lth.cs.nlp.EntityRecognizer.Corpus.NGram;

import java.util.List;

public class NGramDataSet extends MultiDataSet {
    private final List<NGram> nGrams;
    private final INDArray[] features;
    private final INDArray[] labels;

    public NGramDataSet(final List<List<INDArray>> features, final List<INDArray> outputs, final List<NGram> nGrams) {
        this.features = new INDArray[features.size()];

        for (int i = 0; i < features.size(); i++) {
            this.features[i] = Nd4j.vstack(features.get(i));
        }

        // Should only happen during inference
        if (outputs.size() == 0) {
            this.labels = null;
        } else {
            this.labels = new INDArray[]{Nd4j.vstack(outputs)};
        }

        this.nGrams = nGrams;
    }


    public List<NGram> getNGrams() {
        return nGrams;
    }

    @Override
    public INDArray[] getLabels() {
        return labels;
    }

    @Override
    public INDArray[] getFeatures() {
        return features;
    }
}
