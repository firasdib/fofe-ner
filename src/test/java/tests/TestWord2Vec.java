package tests;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Word2Vec;

import static org.junit.Assert.assertEquals;

public class TestWord2Vec {

    @Before
    public void runBefore() {
        Config.readFile("./config.json");
        Nd4j.zeros(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void ensureNormalized() {
        final Word2Vec w2v = new Word2Vec(Config.getString("word2vecModelCS"), Config.getString("word2vecModelCIS"));

        for (int i = 0; i < w2v.getWeights(true).rows(); i++) {
            final INDArray wordVectorCS = w2v.getWeights(true).getRow(i);
            assertEquals(1.0, (double) wordVectorCS.norm2Number(), 1E-5);

            final INDArray wordVectorCIS = w2v.getWeights(false).getRow(i);
            assertEquals(1.0, (double) wordVectorCIS.norm2Number(), 1E-5);
        }
    }

}
