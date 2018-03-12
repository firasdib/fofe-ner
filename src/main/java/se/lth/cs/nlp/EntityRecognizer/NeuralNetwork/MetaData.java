package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork;

import se.lth.cs.nlp.EntityRecognizer.Corpus.Corpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.EntityType;

import java.util.List;

public class MetaData {
    private final Corpus corpus;
    private final Word2Vec w2v;
    private final NetworkMode networkMode;

    public MetaData(final Corpus corpus, final Word2Vec w2v, final NetworkMode networkMode) {
        this.corpus = corpus;
        this.w2v = w2v;
        this.networkMode = networkMode;
    }

    public Corpus getCorpus() {
        return corpus;
    }

    public Word2Vec getW2v() {
        return w2v;
    }

    public NetworkMode getNetworkMode() {
        return networkMode;
    }

    public List<String> getEntityTypes() {
        return EntityType.getEntityTypes(networkMode);
    }
}
