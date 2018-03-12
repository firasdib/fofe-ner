package se.lth.cs.nlp.EntityRecognizer.CLI;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.NetworkMode;

@SuppressWarnings("CanBeFinal")
@Parameters(commandDescription = "Train a neural network")
public class TrainCommand {
    @Parameter(names = {"-c", "--config"}, description = "Specify configuration file", order = 1)
    public String configFile = "./config.json";

    @Parameter(names = {"-s", "--save"}, description = "Save model during training", order = 3)
    public String saveModelName = null;

    @Parameter(names = {"-ui", "--use-ui"}, description = "Use the web UI (note: impacts performance).", order = 4)
    public boolean useStatsUI = false;

    @Parameter(names = {"-m", "--mode"}, description = "Define which type to expect the training and evaluation file to be in.", required = true, order = 5)
    public NetworkMode networkMode = null;

    @Parameter(names = {"-t", "--train"}, description = "Specify file to train on", required = true, order = 6)
    public String trainingFile = null;

    @Parameter(names = {"-e", "--eval"}, description = "Specify file to evaluate on after network has finished training", order = 7)
    public String evaluationFile = null;

    @Parameter(names = {"-d", "--dev-eval"}, description = "Specify file to periodically evaluate on during training", order = 8)
    public String devEvaluationFile = null;
}
