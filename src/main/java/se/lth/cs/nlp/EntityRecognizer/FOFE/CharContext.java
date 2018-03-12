package se.lth.cs.nlp.EntityRecognizer.FOFE;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.Corpus.NGram;

import java.util.Arrays;

public class CharContext {
    public static final int CHAR_DIMENSIONS = Config.getInt("charDimensions");

    public enum Direction {
        LEFT,
        RIGHT
    }

    public static String getAllChars(final NGram nGram, final Direction direction) {
        final StringBuilder words = new StringBuilder();

        nGram.getWords().forEach(word ->
                words.append(word.getWordString()).append(" "));

        words.setLength(words.length() - 1);

        return direction == Direction.LEFT
                ? words.toString()
                : words.reverse().toString();
    }

    public static String getInitialChars(final NGram nGram, final Direction direction) {
        final StringBuilder initials = new StringBuilder();

        nGram.getWords().forEach(word ->
                initials.append(word.getWordString().charAt(0)));

        return direction == Direction.LEFT
                ? initials.toString()
                : initials.reverse().toString();
    }

    public static INDArray create(final NGram nGram, final Direction direction) {
        return create(getAllChars(nGram, direction));
    }

    public static INDArray createInitial(final NGram nGram, final Direction direction) {
        return create(getInitialChars(nGram, direction));
    }

    private static INDArray create(final String string) {
        final double[] encoding = new double[FOFE.MAX_CHAR_LENGTH * 2];
        Arrays.fill(encoding, -1);

        final int loopTo = Math.min(string.length(), FOFE.MAX_CHAR_LENGTH);

        int encIdx = 0;
        for (int i = 0; i < loopTo; i++) {
            final int charIdx = ((int) string.charAt(i)) % CHAR_DIMENSIONS;
            encoding[encIdx] = charIdx;
            encoding[encIdx + 1] = FOFE.CHAR_ALPHAS[i];

            encIdx += 2;
        }

        return Nd4j.create(encoding);
    }
}
