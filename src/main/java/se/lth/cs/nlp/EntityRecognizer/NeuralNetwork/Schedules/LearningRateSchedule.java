package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Schedules;

import lombok.Data;
import org.nd4j.linalg.schedule.ISchedule;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;

@Data
public class LearningRateSchedule implements ISchedule {
    private final boolean useLRDecay = Config.getBool("useLRDecay");
    private final double initialRate;
    private final int epochCount;

    public LearningRateSchedule() {
        // Default values to keep JSON serialization happy, these are unused.
        this.initialRate = 0.1024;
        this.epochCount = 128;
    }

    public LearningRateSchedule(final double initialRate, final int epochCount) {
        this.initialRate = initialRate;
        this.epochCount = epochCount;
    }

    @Override
    public double valueAt(final int iteration, final int epoch) {
        if (!useLRDecay) {
            return initialRate;
        }

        return initialRate * Math.pow(0.5, 4.0 * epoch / epochCount);
    }

    @Override
    @SuppressWarnings("MethodDoesntCallSuperMethod")
    public ISchedule clone() {
        return new LearningRateSchedule(initialRate, epochCount);
    }
}
