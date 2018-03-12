package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork;

import java.util.concurrent.TimeUnit;

public class NNUtils {
    public static String formatTime(final long millis) {
        return String.format("%02d hours, %02d min, %02d sec",
                TimeUnit.MILLISECONDS.toHours(millis),
                TimeUnit.MILLISECONDS.toMinutes(millis) -
                        TimeUnit.HOURS.toMinutes(TimeUnit.MILLISECONDS.toHours(millis)),
                TimeUnit.MILLISECONDS.toSeconds(millis) -
                        TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(millis))
        );
    }
}
