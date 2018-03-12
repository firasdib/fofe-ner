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
import se.lth.cs.nlp.EntityRecognizer.FOFE.WordContext;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Word2Vec;


import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class TestWordContext {

    private static EntitySentence entitySentence;
    private static EntitySentence chineseEntitySentence;
    private static Word2Vec w2v;
    private static double alpha;

    @BeforeClass
    public static void before() {
        Config.readFile("./config.json");
        Config.setAttr("chineseMode", "false");

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


        final List<Word> chineseWords = new ArrayList<>();
        final List<NGram> chineseNGrams = new ArrayList<>();

        chineseWords.add(new CoNLL2002Word("基本 X"));
        chineseWords.add(new CoNLL2002Word("就 X"));
        chineseWords.add(new CoNLL2002Word("能 X"));
        chineseWords.add(new CoNLL2002Word("断定 X"));
        chineseWords.add(new CoNLL2002Word("谁 X"));
        chineseWords.add(new CoNLL2002Word("是 X"));
        chineseWords.add(new CoNLL2002Word("背后 X"));
        chineseWords.add(new CoNLL2002Word("策划者 X"));
        chineseWords.add(new CoNLL2002Word("了 X"));
        chineseWords.add(new CoNLL2002Word("。 X"));

        final List<Word> nGramWords2 = new ArrayList<>();
        nGramWords2.add(chineseWords.get(5));
        nGramWords2.add(chineseWords.get(6));
        chineseNGrams.add(new NGram(nGramWords2, "X", 5, 7));

        final Sentence chineseSentence = new Sentence(chineseWords, chineseNGrams);
        chineseEntitySentence = new EntitySentence(chineseSentence, chineseNGrams.get(0));

        w2v = new Word2Vec(Config.getString("word2vecModelCS"), Config.getString("word2vecModelCIS"));

        alpha = Config.getDouble("wordAlpha");
    }

    @Test
    public void testLeftContextExcl() {
        final INDArray encoding = WordContext.create(entitySentence, w2v, true, false, WordContext.Direction.LEFT);

        assertEquals(w2v.indexOf(w2v.getBos(true), true), encoding.getDouble(0, 0), 1E-5);
        assertEquals(Math.pow(alpha, 0), encoding.getDouble(0, 1), 1E-5);
        assertEquals(-1.0, encoding.getDouble(0, 2), 1E-5);
    }

    @Test
    public void testLeftContextIncl() {
        final INDArray encoding = WordContext.create(entitySentence, w2v, true, true, WordContext.Direction.LEFT);

        final List<Word> words = entitySentence.getSentence().getWords();

        assertEquals(w2v.indexOf(words.get(1).getWordString(), true), encoding.getDouble(0, 0), 1E-5);
        assertEquals(Math.pow(alpha, 0), encoding.getDouble(0, 1), 1E-5);

        assertEquals(w2v.indexOf(words.get(0).getWordString(), true), encoding.getDouble(0, 2), 1E-5);
        assertEquals(Math.pow(alpha, 1), encoding.getDouble(0, 3), 1E-5);

        assertEquals(w2v.indexOf(w2v.getBos(true), true), encoding.getDouble(0, 4), 1E-5);
        assertEquals(Math.pow(alpha, 2), encoding.getDouble(0, 5), 1E-5);

        assertEquals(-1.0, encoding.getDouble(0, 6), 1E-5);
    }

    @Test
    public void testRightContextExcl() {
        final INDArray encoding = WordContext.create(entitySentence, w2v, true, false, WordContext.Direction.RIGHT);

        final List<Word> words = entitySentence.getSentence().getWords();

        assertEquals(w2v.indexOf(words.get(2).getWordString(), true), encoding.getDouble(0, 0), 1E-5);
        assertEquals(Math.pow(alpha, 0), encoding.getDouble(0, 1), 1E-5);

        assertEquals(w2v.indexOf(words.get(3).getWordString(), true), encoding.getDouble(0, 2), 1E-5);
        assertEquals(Math.pow(alpha, 1), encoding.getDouble(0, 3), 1E-5);

        assertEquals(w2v.indexOf(w2v.getEos(true), true), encoding.getDouble(0, 4), 1E-5);
        assertEquals(Math.pow(alpha, 2), encoding.getDouble(0, 5), 1E-5);

        assertEquals(-1.0, encoding.getDouble(0, 6), 1E-5);
    }

    @Test
    public void testRightContextIncl() {
        final INDArray encoding = WordContext.create(entitySentence, w2v, true, true, WordContext.Direction.RIGHT);

        final List<Word> words = entitySentence.getSentence().getWords();

        assertEquals(w2v.indexOf(words.get(0).getWordString(), true), encoding.getDouble(0, 0), 1E-5);
        assertEquals(Math.pow(alpha, 0), encoding.getDouble(0, 1), 1E-5);

        assertEquals(w2v.indexOf(words.get(1).getWordString(), true), encoding.getDouble(0, 2), 1E-5);
        assertEquals(Math.pow(alpha, 1), encoding.getDouble(0, 3), 1E-5);

        assertEquals(w2v.indexOf(words.get(2).getWordString(), true), encoding.getDouble(0, 4), 1E-5);
        assertEquals(Math.pow(alpha, 2), encoding.getDouble(0, 5), 1E-5);

        assertEquals(w2v.indexOf(words.get(3).getWordString(), true), encoding.getDouble(0, 6), 1E-5);
        assertEquals(Math.pow(alpha, 3), encoding.getDouble(0, 7), 1E-5);

        assertEquals(w2v.indexOf(w2v.getEos(true), true), encoding.getDouble(0, 8), 1E-5);
        assertEquals(Math.pow(alpha, 4), encoding.getDouble(0, 9), 1E-5);

        assertEquals(-1.0, encoding.getDouble(0, 10), 1E-5);
    }

    @Test
    public void testChineseModeLeft() {
        Config.setAttr("chineseMode", "true");

        final INDArray encoding = WordContext.create(chineseEntitySentence, w2v, false, true, WordContext.Direction.LEFT);

        assertEquals(w2v.indexOf("基", false), encoding.getDouble(0, 18), 1E-5);
        assertEquals(Math.pow(alpha, 9), encoding.getDouble(0, 19), 1E-5);

        assertEquals(w2v.indexOf("本", false), encoding.getDouble(0, 16), 1E-5);
        assertEquals(Math.pow(alpha, 8), encoding.getDouble(0, 17), 1E-5);

        assertEquals(w2v.indexOf("就", false), encoding.getDouble(0, 14), 1E-5);
        assertEquals(Math.pow(alpha, 7), encoding.getDouble(0, 15), 1E-5);

        assertEquals(w2v.indexOf("能", false), encoding.getDouble(0, 12), 1E-5);
        assertEquals(Math.pow(alpha, 6), encoding.getDouble(0, 13), 1E-5);

        assertEquals(w2v.indexOf("断", false), encoding.getDouble(0, 10), 1E-5);
        assertEquals(Math.pow(alpha, 5), encoding.getDouble(0, 11), 1E-5);

        assertEquals(w2v.indexOf("定", false), encoding.getDouble(0, 8), 1E-5);
        assertEquals(Math.pow(alpha, 4), encoding.getDouble(0, 9), 1E-5);

        assertEquals(w2v.indexOf("谁", false), encoding.getDouble(0, 6), 1E-5);
        assertEquals(Math.pow(alpha, 3), encoding.getDouble(0, 7), 1E-5);

        assertEquals(w2v.indexOf("是", false), encoding.getDouble(0, 4), 1E-5);
        assertEquals(Math.pow(alpha, 2), encoding.getDouble(0, 5), 1E-5);

        assertEquals(w2v.indexOf("背", false), encoding.getDouble(0, 2), 1E-5);
        assertEquals(Math.pow(alpha, 1), encoding.getDouble(0, 3), 1E-5);

        assertEquals(w2v.indexOf("后", false), encoding.getDouble(0, 0), 1E-5);
        assertEquals(Math.pow(alpha, 0), encoding.getDouble(0, 1), 1E-5);

        assertEquals(w2v.indexOf(w2v.getBos(false), false), encoding.getDouble(0, 20), 1E-5);
        assertEquals(Math.pow(alpha, 10), encoding.getDouble(0, 21), 1E-5);

        assertEquals(-1.0, encoding.getDouble(0, 22), 1E-5);

        Config.setAttr("chineseMode", "false");
    }

    @Test
    public void testChineseModeLeftExcl() {
        Config.setAttr("chineseMode", "true");

        final INDArray encoding = WordContext.create(chineseEntitySentence, w2v, false, false, WordContext.Direction.LEFT);

        assertEquals(w2v.indexOf("基", false), encoding.getDouble(0, 12), 1E-5);
        assertEquals(Math.pow(alpha, 6), encoding.getDouble(0, 13), 1E-5);

        assertEquals(w2v.indexOf("本", false), encoding.getDouble(0, 10), 1E-5);
        assertEquals(Math.pow(alpha, 5), encoding.getDouble(0, 11), 1E-5);

        assertEquals(w2v.indexOf("就", false), encoding.getDouble(0, 8), 1E-5);
        assertEquals(Math.pow(alpha, 4), encoding.getDouble(0, 9), 1E-5);

        assertEquals(w2v.indexOf("能", false), encoding.getDouble(0, 6), 1E-5);
        assertEquals(Math.pow(alpha, 3), encoding.getDouble(0, 7), 1E-5);

        assertEquals(w2v.indexOf("断", false), encoding.getDouble(0, 4), 1E-5);
        assertEquals(Math.pow(alpha, 2), encoding.getDouble(0, 5), 1E-5);

        assertEquals(w2v.indexOf("定", false), encoding.getDouble(0, 2), 1E-5);
        assertEquals(Math.pow(alpha, 1), encoding.getDouble(0, 3), 1E-5);

        assertEquals(w2v.indexOf("谁", false), encoding.getDouble(0, 0), 1E-5);
        assertEquals(Math.pow(alpha, 0), encoding.getDouble(0, 1), 1E-5);

        assertEquals(w2v.indexOf(w2v.getBos(false), false), encoding.getDouble(0, 14), 1E-5);
        assertEquals(Math.pow(alpha, 7), encoding.getDouble(0, 15), 1E-5);

        assertEquals(-1.0, encoding.getDouble(0, 16), 1E-5);

        Config.setAttr("chineseMode", "false");
    }

    @Test
    public void testChineseModeLeftWord() {
        Config.setAttr("chineseMode", "true");

        final INDArray encoding = WordContext.create(chineseEntitySentence, w2v, true, true, WordContext.Direction.LEFT);

        assertEquals(w2v.indexOf("基本", true), encoding.getDouble(0, 12), 1E-5);
        assertEquals(Math.pow(alpha, 6), encoding.getDouble(0, 13), 1E-5);

        assertEquals(w2v.indexOf("就", true), encoding.getDouble(0, 10), 1E-5);
        assertEquals(Math.pow(alpha, 5), encoding.getDouble(0, 11), 1E-5);

        assertEquals(w2v.indexOf("能", true), encoding.getDouble(0, 8), 1E-5);
        assertEquals(Math.pow(alpha, 4), encoding.getDouble(0, 9), 1E-5);

        assertEquals(w2v.indexOf("断定", true), encoding.getDouble(0, 6), 1E-5);
        assertEquals(Math.pow(alpha, 3), encoding.getDouble(0, 7), 1E-5);

        assertEquals(w2v.indexOf("谁", true), encoding.getDouble(0, 4), 1E-5);
        assertEquals(Math.pow(alpha, 2), encoding.getDouble(0, 5), 1E-5);

        assertEquals(w2v.indexOf("是", true), encoding.getDouble(0, 2), 1E-5);
        assertEquals(Math.pow(alpha, 1), encoding.getDouble(0, 3), 1E-5);

        assertEquals(w2v.indexOf("背后", true), encoding.getDouble(0, 0), 1E-5);
        assertEquals(Math.pow(alpha, 0), encoding.getDouble(0, 1), 1E-5);

        assertEquals(w2v.indexOf(w2v.getBos(true), true), encoding.getDouble(0, 14), 1E-5);
        assertEquals(Math.pow(alpha, 7), encoding.getDouble(0, 15), 1E-5);

        assertEquals(-1.0, encoding.getDouble(0, 16), 1E-5);

        Config.setAttr("chineseMode", "false");
    }

    @Test
    public void testChineseModeLeftWordExcl() {
        Config.setAttr("chineseMode", "true");

        final INDArray encoding = WordContext.create(chineseEntitySentence, w2v, true, false, WordContext.Direction.LEFT);

        assertEquals(w2v.indexOf("基本", true), encoding.getDouble(0, 8), 1E-5);
        assertEquals(Math.pow(alpha, 4), encoding.getDouble(0, 9), 1E-5);

        assertEquals(w2v.indexOf("就", true), encoding.getDouble(0, 6), 1E-5);
        assertEquals(Math.pow(alpha, 3), encoding.getDouble(0, 7), 1E-5);

        assertEquals(w2v.indexOf("能", true), encoding.getDouble(0, 4), 1E-5);
        assertEquals(Math.pow(alpha, 2), encoding.getDouble(0, 5), 1E-5);

        assertEquals(w2v.indexOf("断定", true), encoding.getDouble(0, 2), 1E-5);
        assertEquals(Math.pow(alpha, 1), encoding.getDouble(0, 3), 1E-5);

        assertEquals(w2v.indexOf("谁", true), encoding.getDouble(0, 0), 1E-5);
        assertEquals(Math.pow(alpha, 0), encoding.getDouble(0, 1), 1E-5);

        assertEquals(w2v.indexOf(w2v.getBos(true), true), encoding.getDouble(0, 10), 1E-5);
        assertEquals(Math.pow(alpha, 5), encoding.getDouble(0, 11), 1E-5);

        assertEquals(-1.0, encoding.getDouble(0, 12), 1E-5);

        Config.setAttr("chineseMode", "false");
    }

}
