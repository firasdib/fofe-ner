package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork;

import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.Features.*;

public class NNFeatures {

    public static FeatureSet create(final Word2Vec w2v) {
        final boolean chineseMode = Config.getBool("chineseMode");
        final boolean useCharFOFEFeatures = Config.getBool("useCharFOFEFeatures");
        final boolean useCharCNNFeatures = Config.getBool("useCharCNNFeatures");
        final boolean useWordFOFEFeaturesCIS = Config.getBool("useWordFOFEFeaturesCIS");
        final boolean useWordFOFEFeaturesCS = Config.getBool("useWordFOFEFeaturesCS");
        final boolean useInitialCharFOFEFeatures = Config.getBool("useInitialCharFOFEFeatures");

        // https://github.com/xmb-cipher/fofe-ner/blob/master/fofe_mention_net.py#L1590
        final FeatureSet fs = new FeatureSet(w2v);

        if (useWordFOFEFeaturesCS) {
            fs.addFeature(new LeftWordFOFE(true, true))
                    .addFeature(new LeftWordFOFE(true, false))
                    .addFeature(new RightWordFOFE(true, true))
                    .addFeature(new RightWordFOFE(true, false))
                    .addFeature(new BagOfWords(true));
        }

        if (useWordFOFEFeaturesCIS) {
            fs.addFeature(new LeftWordFOFE(false, true))
                    .addFeature(new LeftWordFOFE(false, false))
                    .addFeature(new RightWordFOFE(false, true))
                    .addFeature(new RightWordFOFE(false, false))
                    .addFeature(new BagOfWords(false));
        }

        if (useCharFOFEFeatures && !chineseMode) {
            fs.addFeature(new LeftCharFOFE())
                    .addFeature(new RightCharFOFE());
        }

        if (useInitialCharFOFEFeatures && !chineseMode) {
            fs.addFeature(new LeftInitialCharFOFE())
                    .addFeature(new RightInitialCharFOFE());
        }

        if (useCharCNNFeatures && !chineseMode) {
            fs.addFeature(new CharConvolution());
        }

        return fs;
    }
}
