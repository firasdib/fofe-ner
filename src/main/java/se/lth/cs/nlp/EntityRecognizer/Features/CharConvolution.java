package se.lth.cs.nlp.EntityRecognizer.Features;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import se.lth.cs.nlp.EntityRecognizer.Data.EntitySentence;
import se.lth.cs.nlp.EntityRecognizer.FOFE.CharContext;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Word2Vec;

public class CharConvolution extends Feature {
    public CharConvolution() {
        super(FeatureType.CHAR_CNN, true, false);
    }

    @Override
    public INDArray createFeature(final Word2Vec w2v, final EntitySentence entitySentence) {
        final String allChars = CharContext.getAllChars(entitySentence.getNGram(), CharContext.Direction.LEFT);
        final double[][] encoding = new double[100][CharContext.CHAR_DIMENSIONS];

        for (int i = 0; i < Math.min(encoding.length, allChars.length()); i++) {
            final int charIndex = ((int) allChars.charAt(i)) % CharContext.CHAR_DIMENSIONS;

            encoding[i][charIndex] = 1;
        }

        return Nd4j.create(encoding)
                .reshape(1, 1, encoding.length, encoding[0].length);
    }

    @Override
    protected String name() {
        return "charConvolution";
    }
}
