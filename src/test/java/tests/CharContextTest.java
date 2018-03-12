package tests;

import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.Corpus.CoNLL2002.CoNLL2002Word;
import se.lth.cs.nlp.EntityRecognizer.Corpus.NGram;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Sentence;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Word;
import se.lth.cs.nlp.EntityRecognizer.Data.EntitySentence;
import se.lth.cs.nlp.EntityRecognizer.FOFE.CharContext;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class CharContextTest {

    private static EntitySentence entitySentence;
    private static double alpha;

    @BeforeClass
    public static void before() {
        Config.readFile("./config.json");
        Nd4j.zeros(1);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        final List<Word> words = new ArrayList<>();
        final List<NGram> nGrams = new ArrayList<>();

        words.add(new CoNLL2002Word("Hello X"));
        words.add(new CoNLL2002Word("there X"));
        words.add(new CoNLL2002Word("friend X"));
        words.add(new CoNLL2002Word(". X"));

        final List<Word> nGramWords1 = new ArrayList<>();
        nGramWords1.add(words.get(0));
        nGramWords1.add(words.get(1));
        nGrams.add(new NGram(nGramWords1, "X", 0, 2));

        final Sentence sentence = new Sentence(words, nGrams);

        entitySentence = new EntitySentence(sentence, nGrams.get(0));

        alpha = Config.getDouble("charAlpha");
    }

    @Test
    public void testGetAllChars() {
        final String allLeftChars = CharContext.getAllChars(entitySentence.getNGram(), CharContext.Direction.LEFT);
        final String allRightChars = CharContext.getAllChars(entitySentence.getNGram(), CharContext.Direction.RIGHT);

        assertEquals("Hello there", allLeftChars);
        assertEquals("ereht olleH", allRightChars);
    }

    @Test
    public void testGetInitialChars() {
        final String allLeftChars = CharContext.getInitialChars(entitySentence.getNGram(), CharContext.Direction.LEFT);
        final String allRightChars = CharContext.getInitialChars(entitySentence.getNGram(), CharContext.Direction.RIGHT);

        assertEquals("Ht", allLeftChars);
        assertEquals("tH", allRightChars);
    }

    @Test
    public void testLeftCharContext() {
        final INDArray encoding = CharContext.create(entitySentence.getNGram(), CharContext.Direction.LEFT);

        assertEquals((int) 'H', encoding.getDouble(0, 0), 1E-5);
        assertEquals(Math.pow(alpha, 0), encoding.getDouble(0, 1), 1E-5);

        assertEquals((int) 'e', encoding.getDouble(0, 2), 1E-5);
        assertEquals(Math.pow(alpha, 1), encoding.getDouble(0, 3), 1E-5);

        assertEquals((int) 'l', encoding.getDouble(0, 4), 1E-5);
        assertEquals(Math.pow(alpha, 2), encoding.getDouble(0, 5), 1E-5);

        assertEquals((int) 'l', encoding.getDouble(0, 6), 1E-5);
        assertEquals(Math.pow(alpha, 3), encoding.getDouble(0, 7), 1E-5);

        assertEquals((int) 'o', encoding.getDouble(0, 8), 1E-5);
        assertEquals(Math.pow(alpha, 4), encoding.getDouble(0, 9), 1E-5);

        assertEquals((int) ' ', encoding.getDouble(0, 10), 1E-5);
        assertEquals(Math.pow(alpha, 5), encoding.getDouble(0, 11), 1E-5);

        assertEquals((int) 't', encoding.getDouble(0, 12), 1E-5);
        assertEquals(Math.pow(alpha, 6), encoding.getDouble(0, 13), 1E-5);

        assertEquals((int) 'h', encoding.getDouble(0, 14), 1E-5);
        assertEquals(Math.pow(alpha, 7), encoding.getDouble(0, 15), 1E-5);

        assertEquals((int) 'e', encoding.getDouble(0, 16), 1E-5);
        assertEquals(Math.pow(alpha, 8), encoding.getDouble(0, 17), 1E-5);

        assertEquals((int) 'r', encoding.getDouble(0, 18), 1E-5);
        assertEquals(Math.pow(alpha, 9), encoding.getDouble(0, 19), 1E-5);

        assertEquals((int) 'e', encoding.getDouble(0, 20), 1E-5);
        assertEquals(Math.pow(alpha, 10), encoding.getDouble(0, 21), 1E-5);

        assertEquals(-1.0, encoding.getDouble(0, 22), 1E-5);
    }

    @Test
    public void testRightCharContext() {
        final INDArray encoding = CharContext.create(entitySentence.getNGram(), CharContext.Direction.RIGHT);

        assertEquals((int) 'H', encoding.getDouble(0, 20), 1E-5);
        assertEquals(Math.pow(alpha, 10), encoding.getDouble(0, 21), 1E-5);

        assertEquals((int) 'e', encoding.getDouble(0, 18), 1E-5);
        assertEquals(Math.pow(alpha, 9), encoding.getDouble(0, 19), 1E-5);

        assertEquals((int) 'l', encoding.getDouble(0, 16), 1E-5);
        assertEquals(Math.pow(alpha, 8), encoding.getDouble(0, 17), 1E-5);

        assertEquals((int) 'l', encoding.getDouble(0, 14), 1E-5);
        assertEquals(Math.pow(alpha, 7), encoding.getDouble(0, 15), 1E-5);

        assertEquals((int) 'o', encoding.getDouble(0, 12), 1E-5);
        assertEquals(Math.pow(alpha, 6), encoding.getDouble(0, 13), 1E-5);

        assertEquals((int) ' ', encoding.getDouble(0, 10), 1E-5);
        assertEquals(Math.pow(alpha, 5), encoding.getDouble(0, 11), 1E-5);

        assertEquals((int) 't', encoding.getDouble(0, 8), 1E-5);
        assertEquals(Math.pow(alpha, 4), encoding.getDouble(0, 9), 1E-5);

        assertEquals((int) 'h', encoding.getDouble(0, 6), 1E-5);
        assertEquals(Math.pow(alpha, 3), encoding.getDouble(0, 7), 1E-5);

        assertEquals((int) 'e', encoding.getDouble(0, 4), 1E-5);
        assertEquals(Math.pow(alpha, 2), encoding.getDouble(0, 5), 1E-5);

        assertEquals((int) 'r', encoding.getDouble(0, 2), 1E-5);
        assertEquals(Math.pow(alpha, 1), encoding.getDouble(0, 3), 1E-5);

        assertEquals((int) 'e', encoding.getDouble(0, 0), 1E-5);
        assertEquals(Math.pow(alpha, 0), encoding.getDouble(0, 1), 1E-5);

        assertEquals(-1.0, encoding.getDouble(0, 22), 1E-5);
    }

    @Test
    public void testLeftInitialCharContext() {
        final INDArray encoding = CharContext.createInitial(entitySentence.getNGram(), CharContext.Direction.LEFT);

        assertEquals((int) 'H', encoding.getDouble(0, 0), 1E-5);
        assertEquals(Math.pow(alpha, 0), encoding.getDouble(0, 1), 1E-5);

        assertEquals((int) 't', encoding.getDouble(0, 2), 1E-5);
        assertEquals(Math.pow(alpha, 1), encoding.getDouble(0, 3), 1E-5);

        assertEquals(-1.0, encoding.getDouble(0, 4), 1E-5);
    }

    @Test
    public void testRightInitialCharContext() {
        final INDArray encoding = CharContext.createInitial(entitySentence.getNGram(), CharContext.Direction.RIGHT);

        assertEquals((int) 't', encoding.getDouble(0, 0), 1E-5);
        assertEquals(Math.pow(alpha, 0), encoding.getDouble(0, 1), 1E-5);

        assertEquals((int) 'H', encoding.getDouble(0, 2), 1E-5);
        assertEquals(Math.pow(alpha, 1), encoding.getDouble(0, 3), 1E-5);

        assertEquals(-1.0, encoding.getDouble(0, 4), 1E-5);
    }
}
