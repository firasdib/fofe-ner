package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Schedules;

import lombok.Data;
import org.nd4j.linalg.schedule.ISchedule;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;

@Data
public class DropoutSchedule implements ISchedule {
    private final boolean useDropoutDecay = Config.getBool("useDropoutDecay");
    private final double initialRate;
    private final int epochCount;

    public DropoutSchedule() {
        // Default values to keep JSON serialization happy, these are unused.
        this.initialRate = 0.4096;
        this.epochCount = 128;
    }

    public DropoutSchedule(final double initialRate, final int epochCount) {
        this.initialRate = initialRate;
        this.epochCount = epochCount;
    }

    @Override
    public double valueAt(final int iteration, final int epoch) {
        if (!useDropoutDecay) {
            return 1 - initialRate;
        }

        return 1 - initialRate * Math.pow(0.5, 2.0 * epoch / epochCount);
    }

    @Override
    @SuppressWarnings("MethodDoesntCallSuperMethod")
    public ISchedule clone() {
        return new DropoutSchedule(initialRate, epochCount);
    }
}
