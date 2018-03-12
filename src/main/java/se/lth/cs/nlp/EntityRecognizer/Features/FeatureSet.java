package se.lth.cs.nlp.EntityRecognizer.Features;

import org.nd4j.linalg.api.ndarray.INDArray;
import se.lth.cs.nlp.EntityRecognizer.Data.EntitySentence;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Word2Vec;

import java.util.*;

public class FeatureSet {
    private final Word2Vec w2v;
    private final List<Feature> features = new ArrayList<>();

    public FeatureSet(final Word2Vec w2v) {
        this.w2v = w2v;
    }

    public FeatureSet addFeature(final Feature feature) {
        // Sanity check..
        if (feature.getFeatureType() == FeatureType.CHAR_CNN) {
            features.forEach(existingFeature -> {
                if (existingFeature.getFeatureType() == FeatureType.CHAR_CNN) {
                    throw new RuntimeException("You can not add multiple character convolution features");
                }
            });
        }

        features.add(feature);

        return this;
    }

    public INDArray[] getFeatureVector(final EntitySentence entitySentence) {
        final INDArray[] featureVectors = new INDArray[features.size()];

        int i = 0;
        for (final Feature feature : features) {
            final INDArray featureVector = feature.createFeature(w2v, entitySentence);

            featureVectors[i] = featureVector;

            i++;
        }

        return featureVectors;
    }

    public List<Feature> getFeatures() {
        return features;
    }
}
