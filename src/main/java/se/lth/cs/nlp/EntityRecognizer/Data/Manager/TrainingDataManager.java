package se.lth.cs.nlp.EntityRecognizer.Data.Manager;

import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.Corpus.NGram;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Sentence;
import se.lth.cs.nlp.EntityRecognizer.Data.EntitySentence;
import se.lth.cs.nlp.EntityRecognizer.Data.Iterator.DataIterator;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.MetaData;

import java.text.DecimalFormat;
import java.util.*;

public class TrainingDataManager extends DataManager {
    private static final Logger log = LoggerFactory.getLogger(TrainingDataManager.class);
    private static final Random RNG = new Random(123);

    private final double overlapRate;
    private final double disjointRate;
    private final double positiveRate;
    private final List<EntitySentence> positiveSentences = new ArrayList<>();
    private final List<EntitySentence> disjointSentences = new ArrayList<>();
    private final List<EntitySentence> overlappingSentences = new ArrayList<>();

    public TrainingDataManager(final MetaData metaData) {
        super(metaData);

        overlapRate = Config.getDouble("overlapRate");
        disjointRate = Config.getDouble("disjointRate");
        positiveRate = Config.getDouble("positiveRate");

        if (overlapRate + disjointRate + positiveRate > 1.0) {
            throw new IllegalArgumentException("Overlap, disjoint and positive rate must add up to 1 for a proper distribution to be created");
        }

        createAllEntitySentences(metaData.getCorpus().getSentences());
    }

    private void createAllEntitySentences(final List<Sentence> sentences) {
        for (final Sentence sentence : sentences) {
            for (final NGram nGram : sentence.getAllNGrams()) {
                final EntitySentence es = new EntitySentence(sentence, nGram);

                if (nGram.isOverlapping()) {
                    overlappingSentences.add(es);
                } else if (nGram.isDisjoint()) {
                    disjointSentences.add(es);
                } else {
                    positiveSentences.add(es);
                }
            }
        }
    }

    private List<EntitySentence> getNormalizedSentences() {
        // Shuffle all sentences
        Collections.shuffle(disjointSentences, RNG);
        Collections.shuffle(overlappingSentences, RNG);

        // Calculate total sentence count from the positive part
        final int totalSentenceCount = (int) (positiveSentences.size() / positiveRate);
        final List<EntitySentence> normalizedSentences = new ArrayList<>(totalSentenceCount);

        normalizedSentences.addAll(positiveSentences);

        final int disjointCount = Math.min((int) (totalSentenceCount * disjointRate), disjointSentences.size());
        normalizedSentences.addAll(disjointSentences.subList(0, disjointCount));

        final int overlappingCount = Math.min((int) (totalSentenceCount * overlapRate), overlappingSentences.size());
        normalizedSentences.addAll(overlappingSentences.subList(0, overlappingCount));

        Collections.shuffle(normalizedSentences, RNG);

        final DecimalFormat countFormat = new DecimalFormat("#,###");
        final DecimalFormat decimalFormat = new DecimalFormat("#.00");
        final double sentenceCount = normalizedSentences.size();

        log.debug("Normalized sentences consisting of: {} positive ({}%), {} disjoint ({}%), and {} overlapping ({}%). " +
                "Dropped {} disjoint samples ({}%) and {} overlapping samples ({}%).",
                countFormat.format(positiveSentences.size()),
                decimalFormat.format(100 * (positiveSentences.size() / sentenceCount)),
                countFormat.format(disjointCount),
                decimalFormat.format(100 * (disjointCount / sentenceCount)),
                countFormat.format(overlappingCount),
                decimalFormat.format(100 * (overlappingCount / sentenceCount)),
                countFormat.format(disjointSentences.size() - disjointCount),
                decimalFormat.format(100 * (1 - (disjointCount / (double) disjointSentences.size()))),
                countFormat.format(overlappingSentences.size() - overlappingCount ),
                decimalFormat.format(100 * (1 - (overlappingCount  / (double) overlappingSentences.size()))));

        return normalizedSentences;
    }

    @NotNull
    @Override
    protected MultiDataSetIterator create() {
        return new DataIterator(this.getNormalizedSentences(), featureSet, metaData.getEntityTypes());
    }
}
