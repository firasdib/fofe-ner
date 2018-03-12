package se.lth.cs.nlp.EntityRecognizer.Features;

import org.nd4j.linalg.api.ndarray.INDArray;
import se.lth.cs.nlp.EntityRecognizer.Data.EntitySentence;
import se.lth.cs.nlp.EntityRecognizer.FOFE.CharContext;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Word2Vec;

public class LeftCharFOFE extends Feature {
    public LeftCharFOFE() {
        super(FeatureType.CHAR, true, false);
    }

    @Override
    public INDArray createFeature(final Word2Vec w2v, final EntitySentence entitySentence) {
        return CharContext.create(entitySentence.getNGram(), CharContext.Direction.LEFT);
    }

    @Override
    protected String name() {
        return "leftCharFOFE";
    }
}
