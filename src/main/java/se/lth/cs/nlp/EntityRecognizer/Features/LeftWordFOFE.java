package se.lth.cs.nlp.EntityRecognizer.Features;

import org.nd4j.linalg.api.ndarray.INDArray;
import se.lth.cs.nlp.EntityRecognizer.Data.EntitySentence;
import se.lth.cs.nlp.EntityRecognizer.FOFE.WordContext;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Word2Vec;

public class LeftWordFOFE extends Feature {
    public LeftWordFOFE(final boolean caseSensitive, final boolean includeFocusWords) {
        super(FeatureType.WORD, caseSensitive, includeFocusWords);
    }

    @Override
    public INDArray createFeature(Word2Vec w2v, final EntitySentence entitySentence) {
        return WordContext.create(entitySentence, w2v, caseSensitive, includeFocusWords, WordContext.Direction.LEFT);
    }

    @Override
    public String name() {
        return "leftWordFOFE";
    }
}
