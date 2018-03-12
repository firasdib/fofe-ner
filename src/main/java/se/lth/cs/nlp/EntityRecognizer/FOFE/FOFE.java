package se.lth.cs.nlp.EntityRecognizer.FOFE;

import se.lth.cs.nlp.EntityRecognizer.Config.Config;

public class FOFE {
    public static final double WORD_ALPHA = Config.getDouble("wordAlpha");
    public static final double CHAR_ALPHA = Config.getDouble("charAlpha");

    // We can afford to be quite conservative here.
    // Its seldom we're going to encounter more than 150
    // characters in 7 words. That would be roughly 20 characters per word.
    // The value is chosen arbitrarily, mainly just to improve perf a little bit.

    public static final int MAX_CHAR_LENGTH = calcMaxLength(CHAR_ALPHA);
    public static final int MAX_WORD_LENGTH = calcMaxLength(WORD_ALPHA);

    public static final double[] WORD_ALPHAS = generateAlphas(MAX_WORD_LENGTH, WORD_ALPHA);
    public static final double[] CHAR_ALPHAS = generateAlphas(MAX_CHAR_LENGTH, CHAR_ALPHA);

    public static double[] generateAlphas(final int howMany, final double alpha) {
        final double[] alphas = new double[howMany];

        for (int i = 0; i < alphas.length; i++) {
            alphas[i] = Math.pow(alpha, i);
        }

        return alphas;
    }

    private static int calcMaxLength(final double alpha) {
        // The minimum value (1.175494E-38) is taken from the C++ definition (which nd4j uses).
        // Plus one to handle potential round off errors

        return (int) (Math.log(1E-38) / Math.log(alpha)) + 1;
    }
}
