package se.lth.cs.nlp.EntityRecognizer.Features;

import org.nd4j.linalg.api.ndarray.INDArray;
import se.lth.cs.nlp.EntityRecognizer.Data.EntitySentence;
import se.lth.cs.nlp.EntityRecognizer.FOFE.CharContext;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Word2Vec;

public class RightInitialCharFOFE extends Feature {
    public RightInitialCharFOFE() {
        super(FeatureType.CHAR, true, false);
    }

    @Override
    public INDArray createFeature(final Word2Vec w2v, final EntitySentence entitySentence) {
        return CharContext.createInitial(entitySentence.getNGram(), CharContext.Direction.RIGHT);
    }

    @Override
    protected String name() {
        return "rightInitialCharFOFE";
    }
}
