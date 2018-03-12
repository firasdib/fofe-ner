package se.lth.cs.nlp.EntityRecognizer.Features;

import org.nd4j.linalg.api.ndarray.INDArray;
import se.lth.cs.nlp.EntityRecognizer.Data.EntitySentence;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Word2Vec;

public abstract class Feature {
    private final FeatureType featureType;
    protected final boolean caseSensitive;
    protected final boolean includeFocusWords;

    public Feature(final FeatureType featureType, final boolean caseSensitive, final boolean includeFocusWords) {
        this.featureType = featureType;
        this.caseSensitive = caseSensitive;
        this.includeFocusWords = includeFocusWords;
    }

    public abstract INDArray createFeature(final Word2Vec w2v, final EntitySentence entitySentence);

    protected abstract String name();

    public String getName() {
        final StringBuilder sb = new StringBuilder();

        sb.append(name());

        if (!caseSensitive) {
            sb.append("_lowerCase");
        }

        if (includeFocusWords) {
            sb.append("_inclFocus");
        }

        return sb.toString();
    }

    public FeatureType getFeatureType() {
        return featureType;
    }

    public boolean isCaseSensitive() {
        return caseSensitive;
    }
}
