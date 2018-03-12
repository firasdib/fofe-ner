package se.lth.cs.nlp.EntityRecognizer.CLI;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.NetworkMode;

@SuppressWarnings("CanBeFinal")
@Parameters(commandDescription = "Evaluate a pre-trained neural network model")
public class EvalCommand {
    @Parameter(names = {"-c", "--config"}, description = "Specify configuration file", order = 1)
    public String configFile = "./config.json";

    @Parameter(names = {"-l", "--load"}, description = "Load and evaluate an existing model", required = true, order = 2)
    public String loadModelName = null;

    @Parameter(names = {"-m", "--mode"}, description = "Define which type to expect the training and evaluation file to be in.", required = true, order = 5)
    public NetworkMode networkMode = null;

    @Parameter(names = {"-e", "--eval"}, description = "Specify file to evaluate on", required = true, order = 7)
    public String evaluationFile = null;
}
