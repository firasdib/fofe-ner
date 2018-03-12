package se.lth.cs.nlp.EntityRecognizer.FOFE;

import com.google.common.collect.Lists;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.Corpus.NGram;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Word;
import se.lth.cs.nlp.EntityRecognizer.Data.EntitySentence;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Word2Vec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class WordContext {
    public enum Direction {
        LEFT,
        RIGHT
    }

    public static INDArray create(final EntitySentence entitySentence, final Word2Vec w2v, final boolean caseSensitive,
                                  final boolean includeFocusWords, final Direction direction) {
        final boolean chineseMode = Config.getBool("chineseMode");
        final NGram focusNGram = entitySentence.getNGram();

        final List<Word> originalWords = entitySentence.getSentence().getWords();

        // Begin looping at the n-gram
        int loopFrom;

        if (direction == Direction.LEFT) {
            // Reduce one, since loop below is inclusive of index and endIndex is exclusive
            loopFrom = (includeFocusWords ? focusNGram.getEndIndex() : focusNGram.getStartIndex()) - 1;
        } else {
            // Indices are inclusive of start, exclusive of end. Thus no need for the -1 stuff here.
            loopFrom = (originalWords.size() - 1) - (includeFocusWords ? focusNGram.getStartIndex() : focusNGram.getEndIndex());
        }

        List<String> allWords;

        // Chinese mode needs some extra logic
        if (chineseMode && !caseSensitive) {
            final ChineseConversion cc = new ChineseConversion(originalWords, loopFrom);

            allWords = cc.getSymbols();

            // Override with the symbol index instead
            loopFrom = cc.getLoopFrom();
        } else {
            allWords = originalWords.stream()
                    .map(Word::getWordString)
                    .collect(Collectors.toList());
        }

        if (direction == Direction.RIGHT) {
            allWords = Lists.reverse(allWords);
        }

        // Only grab as many words as we can reasonably represent, the rest will be zero due to loss of precision
        // Max length is -1 since index start from 0
        // Example: A B C D E F G
        //          ^         ^
        //       Loop To    Loop From

        final int loopTo = Math.max(0, loopFrom - (FOFE.MAX_WORD_LENGTH - 1));

        final double[] encoding = new double[FOFE.MAX_WORD_LENGTH * 2];
        Arrays.fill(encoding, -1);

        int fofeIndex = 0;
        int encIndex = 0;

        // Loop from the n-gram to the beginning (or cutoff)
        for (int i = loopFrom; i >= loopTo; i--) {
            final String wordString = allWords.get(i);
            final int wordIndex = w2v.indexOf(wordString, caseSensitive);

            encoding[encIndex] = wordIndex;
            encoding[encIndex + 1] = FOFE.WORD_ALPHAS[fofeIndex];

            encIndex += 2;
            fofeIndex++;
        }

        // We have enough precision to model BOS/EOS
        if (encIndex < encoding.length) {
            final String BOS_EOS = direction == Direction.LEFT ? w2v.getBos(caseSensitive) : w2v.getEos(caseSensitive);

            encoding[encIndex] = w2v.indexOf(BOS_EOS, caseSensitive);
            encoding[encIndex + 1] = FOFE.WORD_ALPHAS[fofeIndex];
        }

        return Nd4j.create(encoding);
    }

    private static class ChineseConversion {
        private final int loopFrom;
        private final List<String> symbols;

        public ChineseConversion(final List<Word> allWords, final int loopFrom) {
            final List<String> allSymbols = new ArrayList<>();

            for (int i = 0; i <= loopFrom; i++) {
                final Word word = allWords.get(i);
                @SuppressWarnings({"SyntaxError", "Annotator"}) final String[] symbols = word.getWordString().split("(?=\\p{script=Han})|(?<=\\p{script=Han})");

                allSymbols.addAll(Arrays.asList(symbols));

            }

            this.symbols = allSymbols;
            this.loopFrom = allSymbols.size() - 1;
        }

        public int getLoopFrom() {
            return loopFrom;
        }

        public List<String> getSymbols() {
            return symbols;
        }
    }
}
