package se.lth.cs.nlp.EntityRecognizer.Server;

import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.NNManager;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.NetworkMode;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.NeuralNetwork;

public class Pipeline {
    private final NeuralNetwork network;
    private final String language;
    private final String name;
    private final String model;
    private final NetworkMode mode;
    private final String config;

    public Pipeline(final String model, final String mode, final String config, final String language, final String name) {
        this.model = model;
        this.mode = NetworkMode.valueOf(mode.toUpperCase());
        this.config = config;
        this.language = language;
        this.name = name;

        Config.readFile(config);
        network = NNManager.load(model, this.mode);
    }

    public NeuralNetwork getNetwork() {
        return network;
    }

    public String getLanguage() {
        return language;
    }

    public String getName() {
        return name;
    }

    public String getModel() {
        return model;
    }

    public NetworkMode getMode() {
        return mode;
    }

    public String getConfig() {
        return config;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Pipeline)) return false;

        final Pipeline pipeline = (Pipeline) o;

        return language.equals(pipeline.language) && name.equals(pipeline.name);
    }

    @Override
    public int hashCode() {
        int result = language.hashCode();
        result = 31 * result + name.hashCode();

        return result;
    }
}
