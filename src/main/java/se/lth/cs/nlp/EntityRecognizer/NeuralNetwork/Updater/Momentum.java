package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Updater;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.IUpdater;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Schedules.LearningRateSchedule;

public class Momentum implements IUpdater {
    public static final double DEFAULT_MOMENTUM = 0.9;
    public static final double DEFAULT_LEARNING_RATE = 0.1;

    private final double momentum;
    private final LearningRateSchedule lrSchedule;

    public Momentum() {
        // Unused but necessary to be able to restore the network
        this(new LearningRateSchedule(DEFAULT_LEARNING_RATE, 128), DEFAULT_MOMENTUM);
    }

    public Momentum(final LearningRateSchedule lrSchedule, final double momentum) {
        this.lrSchedule = lrSchedule;
        this.momentum = momentum;
    }

    @Override
    public long stateSize(final long numParams) {
        return numParams;
    }

    @Override
    public GradientUpdater instantiate(final INDArray viewArray, final boolean initializeViewArray) {
        final MomentumUpdater momentumUpdater = new MomentumUpdater(this);
        momentumUpdater.setStateViewArray(viewArray, viewArray.shape(), viewArray.ordering(), initializeViewArray);

        return momentumUpdater;
    }

    @Override
    @SuppressWarnings("MethodDoesntCallSuperMethod")
    public IUpdater clone() {
        return new Momentum(lrSchedule, momentum);
    }

    @Override
    public double getLearningRate(final int iteration, final int epoch) {
        return lrSchedule.valueAt(iteration, epoch);
    }

    public double getMomentum() {
        return momentum;
    }
}
