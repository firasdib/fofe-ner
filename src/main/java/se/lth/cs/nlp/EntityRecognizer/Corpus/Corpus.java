package se.lth.cs.nlp.EntityRecognizer.Corpus;

import com.google.common.collect.Range;
import com.google.common.collect.RangeSet;
import com.google.common.collect.TreeRangeSet;
import org.jetbrains.annotations.NotNull;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.Corpus.CoNLL2002.CoNLL2002Corpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.CoNLL2003.CoNLL2003Corpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Docforia.DocforiaCorpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.SUC.SUCCorpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.TAC.TACCorpus;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.NetworkMode;

import java.util.*;

public abstract class Corpus implements Iterable<Sentence> {
    private final List<Sentence> sentences = new ArrayList<>();

    private List<NGram> createNGrams(final List<Word> words) {
        final int maxNGramSize = Config.getInt("maxNGramSize");

        // Keep track of all entities and how they span
        final RangeSet<Integer> entityRanges = TreeRangeSet.create();
        final List<NGram> nGrams = new ArrayList<>();
        int wordIndex = 0;

        for (int i = 0; i < words.size(); i++) {
            final int nGramSize = Math.min(maxNGramSize, words.size() - i);

            for (int j = 0; j < nGramSize; j++) {
                // This helps us keep track of which tokens this n-gram spans.
                // From wordIndex to nGramEndWordIndex

                int nGramEndWordIndex = wordIndex;

                /*
                 * This will start with the largest n-gram at this position,
                 * and work its way down.
                 *
                 * In the sentence: A B C D E F G
                 *                  ^
                 * It will produce (A, B, C, D, E)
                 *                 (A, B, C, D)
                 *                 (A, B, C)
                 *                 (A, B)
                 *                 (A)
                 * And so on.
                 */

                final List<Word> nGramWords = new ArrayList<>();
                for (int k = 0; k < (nGramSize - j); k++) {
                    final Word currentWord = words.get(wordIndex + k);

                    nGramWords.add(currentWord);

                    nGramEndWordIndex++;
                }

                if (nGramWords.size() > 0) {
                    /*
                     * The reasoning here is that if we have found an entity of say
                     * size 3 to be a valid entity, then size 2 will also be since it
                     * will be contained by size 3. However, that is not necessarily a
                     * valid entity. "University of Lund" is an entity of ORG, thus
                     * "University", "of", and "Lund" would all have the ORG tag.
                     * However, only all three together produce an ORG.
                     * Neither "University of" or "of Lund" is an ORG.
                     */

                    final Range<Integer> range = Range.closedOpen(wordIndex, nGramEndWordIndex);
                    final boolean isIntersecting = !entityRanges.subRangeSet(range).isEmpty();

                    String entityType;

                    // Any intersection -> NONE
                    if (isIntersecting) {
                        entityType = EntityType.NONE;
                    } else {
                        entityType = getEntityType(nGramWords);

                        // If it does not intersect, make sure we have grabbed the ENTIRE mention
                        // and not just a part of it.
                        // This can happen if the mention is longer than MAX_N_GRAM_SIZE

                        if (!entityType.equals(EntityType.NONE) && nGramEndWordIndex < words.size()) {
                            // Test to see if this nGram is of the same type IF we add the next word
                            final List<Word> nGramProbe = new ArrayList<>(nGramWords);
                            nGramProbe.add(words.get(nGramEndWordIndex));

                            // If they are the same entity this means we have not grabbed the entire entity
                            // and this is out of scope for our system.
                            if (entityType.equals(getEntityType(nGramProbe))) {
                                entityType = EntityType.NONE;
                            }
                        }
                    }

                    if (!isIntersecting && !entityType.equals(EntityType.NONE)) {
                        entityRanges.add(range);
                    }

                    nGrams.add(new NGram(nGramWords, entityType, wordIndex, nGramEndWordIndex));
                }
            }

            wordIndex++;
        }

        return nGrams;
    }

    public void addSentence(final List<Word> words) {
        final List<NGram> nGramList = createNGrams(words);

        sentences.add(new Sentence(words, nGramList));
    }

    public List<Sentence> getSentences() {
        return sentences;
    }

    @NotNull
    @Override
    public Iterator<Sentence> iterator() {
        return sentences.iterator();
    }

    public static Corpus createCorpus(final NetworkMode networkMode) {
        return createCorpus(networkMode, null);
    }

    public static Corpus createCorpus(final NetworkMode networkMode, final String corpusFilePath) {
        switch (networkMode) {
            case CONLL2002:
                return new CoNLL2002Corpus(corpusFilePath);
            case CONLL2003:
                return new CoNLL2003Corpus(corpusFilePath);
            case DOCFORIA_TAC:
            case DOCFORIA_CONLL:
            case DOCFORIA_SUC:
                return new DocforiaCorpus();
            case TAC:
                return new TACCorpus(corpusFilePath);
            case SUC:
                return new SUCCorpus(corpusFilePath);
            case SUC_RAW:
                return new SUCCorpus(corpusFilePath, true);
            default:
                throw new RuntimeException("Unknown mode '" + networkMode + "' given.");
        }
    }

    @NotNull
    protected abstract String getEntityType(final List<Word> words);

    public static String getIOB2EntityType(final List<Word> words) {
        /*
         * The first word must start with B-TAG since this is IOB2
         */

        final String firstEntityTag = words.get(0).getEntityTag();

        if (firstEntityTag.equals(EntityType.NONE) || !firstEntityTag.startsWith("B-")) {
            return EntityType.NONE;
        }

        final String entityType = firstEntityTag.substring(2);
        final String IOBEntityType = "I-" + entityType;

        for (int i = 1; i < words.size(); i++) {
            final Word word = words.get(i);

            if (!word.getEntityTag().equals(IOBEntityType)) {
                return EntityType.NONE;
            }
        }

        return entityType;
    }

    public static String getIOBEntityType(final List<Word> words) {
        /*
         * One of the following must be true:
         *
         * 1) The first word has a B-NAME and the other I-NAME
         * 2) All are I-NAME
         */

        if (words.get(0).getEntityTag().equals(EntityType.NONE)) {
            return EntityType.NONE;
        }

        final String entityType = words.get(0).getEntityTag().substring(2);
        final String IOBEntityType = "I-" + entityType;

        for (int i = 1; i < words.size(); i++) {
            final Word word = words.get(i);

            if (!word.getEntityTag().equals(IOBEntityType)) {
                return EntityType.NONE;
            }
        }

        return entityType;
    }
}
