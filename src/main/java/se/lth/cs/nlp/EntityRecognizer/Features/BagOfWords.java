package se.lth.cs.nlp.EntityRecognizer.Features;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Word;
import se.lth.cs.nlp.EntityRecognizer.Data.EntitySentence;
import se.lth.cs.nlp.EntityRecognizer.FOFE.FOFE;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Word2Vec;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class BagOfWords extends Feature {
    public BagOfWords(final boolean caseSensitive) {
        super(FeatureType.WORD, caseSensitive, false);
    }

    @Override
    public INDArray createFeature(final Word2Vec w2v, final EntitySentence entitySentence) {
        final List<String> focusWords = getWordStrings(entitySentence.getNGram().getWords());
        final List<String> allWords = getWordStrings(entitySentence.getSentence().getWords());

        final double[] encoding = new double[FOFE.MAX_WORD_LENGTH * 2];
        Arrays.fill(encoding, -1);

        // BoW should never be large enough to be too large... lets skip these checks for now
        int encIdx = 0;
        for (final String word : focusWords) {
            encoding[encIdx] = w2v.indexOf(word, caseSensitive);
            encoding[encIdx + 1] = countWordFrequency(allWords, word);

            encIdx += 2;
        }

        return Nd4j.create(encoding);
    }

    @Override
    protected String name() {
        return "bagOfWords";
    }

    private int countWordFrequency(final List<String> words, final String word) {
        return (int) words.stream()
                .filter(otherWord -> otherWord.equals(word))
                .count();
    }

    private List<String> getWordStrings(final List<Word> words) {
        return words.stream()
                .map(word -> caseSensitive ? word.getWordString() : word.getWordString().toLowerCase())
                .collect(Collectors.toList());
    }
}
