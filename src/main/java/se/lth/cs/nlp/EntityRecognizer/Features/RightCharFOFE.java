package se.lth.cs.nlp.EntityRecognizer.Features;

import org.nd4j.linalg.api.ndarray.INDArray;
import se.lth.cs.nlp.EntityRecognizer.Data.EntitySentence;
import se.lth.cs.nlp.EntityRecognizer.FOFE.CharContext;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Word2Vec;

public class RightCharFOFE extends Feature {
    public RightCharFOFE() {
        super(FeatureType.CHAR, true, false);
    }

    @Override
    public INDArray createFeature(final Word2Vec w2v, final EntitySentence entitySentence) {
        return CharContext.create(entitySentence.getNGram(), CharContext.Direction.RIGHT);
    }

    @Override
    protected String name() {
        return "rightCharFOFE";
    }
}
