package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public class Word2Vec {
    private static final Logger log = LoggerFactory.getLogger(Word2Vec.class);
    private final boolean chineseMode = Config.getBool("chineseMode");

    private final Map<String, Integer> vocabularyCS = new HashMap<>();
    private final Map<String, Integer> vocabularyCIS = new HashMap<>();
    private final String unkWord;
    private final String unkWordLC;
    private final String bosWord;
    private final String eosWord;
    private INDArray weightsCS = null;
    private INDArray weightsCIS = null;

    public Word2Vec(final String embeddingsCS, final String embeddingsCIS, final boolean parseVocabularyOnly) {
        unkWord = Config.getString("unknownWord");
        unkWordLC = Config.getString("unknownWordLC");
        bosWord = Config.getString("bosWord");
        eosWord = Config.getString("eosWord");

        parseFile(embeddingsCS, true, parseVocabularyOnly);
        parseFile(embeddingsCIS, false, parseVocabularyOnly);

        if (!parseVocabularyOnly) {
            normalizeVectors();
        }
    }

    public Word2Vec(final String embeddingsCS, final String embeddingsCIS) {
        this(embeddingsCS, embeddingsCIS, false);
    }

    private void parseFile(final String file, final boolean caseSensitive, final boolean onlyVocabulary) {
        log.info("Loading word2vec {} from '{}'", onlyVocabulary ? "vocabulary" : "model", file);

        try (final BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {
            final int vocabLimit = Config.getInt("vocabLimit");
            final int wordDim = Config.getInt("dimWordProjection");

            int wordIdx = 0;
            final double[][] weights = onlyVocabulary ? null : new double[vocabLimit][wordDim];

            String line;
            while ((line = br.readLine()) != null) {
                // Limit vocabulary size, assumes the vectors are ordered after frequency, top being first.
                if (wordIdx >= vocabLimit) {
                    break;
                }

                final String[] split = line.split(" ");

                // Add word to the vocabulary
                (caseSensitive ? vocabularyCS : vocabularyCIS).put(split[0], wordIdx);

                // Get the weights
                for (int i = 1; i < split.length && !onlyVocabulary; i++) {
                    // The first token is the word, but the index still needs to be correct, hence "- 1"
                    weights[wordIdx][i - 1] = Double.parseDouble(split[i]);
                }

                wordIdx++;
            }

            if (wordIdx < vocabLimit) {
                throw new RuntimeException("Not enough vectors found in file. Found '" + wordIdx + "' but expected '" + vocabLimit + "'");
            }

            if (!onlyVocabulary) {
                if (caseSensitive) {
                    weightsCS = Nd4j.create(weights);
                } else {
                    weightsCIS = Nd4j.create(weights);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("Unable to parse word2vec file", e);
        }
    }

    public INDArray getWeights(final boolean caseSensitive) {
        return caseSensitive ? weightsCS : weightsCIS;
    }

    private void normalizeVectors() {
        log.info("Normalizing word embeddings...");
        weightsCS.diviColumnVector(weightsCS.norm2(1));
        weightsCIS.diviColumnVector(weightsCIS.norm2(1));

        log.info("Normalization done!");
    }

    public int indexOf(final String word, final boolean caseSensitive) {
        final Map<String, Integer> vocabulary = caseSensitive ? vocabularyCS : vocabularyCIS;
        final String wordString = (caseSensitive || chineseMode) ? word : word.toLowerCase();

        if (!vocabulary.containsKey(wordString)) {
            // For the case-sensitive embedding,
            // an OOV is mapped to <unk> if it contains
            // no upper-case letter and <UNK> otherwise.

            if (!chineseMode && caseSensitive && word.equals(word.toLowerCase())) {
                // This assumes the model has been trained with two unknown vectors;
                // one for each case possibility.

                return vocabularyCS.get(unkWordLC);
            }

            return vocabulary.get(unkWord);
        }

        return vocabulary.get(wordString);
    }

    public String getBos(final boolean caseSensitive) {
        if (!chineseMode && !caseSensitive) {
            return bosWord.toLowerCase();
        }

        return bosWord;
    }

    public String getEos(final boolean caseSensitive) {
        if (!chineseMode && !caseSensitive) {
            return eosWord.toLowerCase();
        }

        return eosWord;
    }
}
