package se.lth.cs.nlp.EntityRecognizer.Corpus;

import java.util.List;

public class NGram {
    private final int startIndex;
    private final int endIndex;
    private final List<Word> words;
    private boolean disjoint = false;
    private boolean overlap = false;
    private double probability;
    private String predictedEntityTag = null;
    private boolean survivedPruning = false;
    private String entityType = null;

    public NGram(final List<Word> words, final String entityType, final int startIndex, final int endIndex) {
        this.words = words;
        this.entityType = entityType;
        this.startIndex = startIndex;
        this.endIndex = endIndex;

        // If this n-gram is of NONE type, find out where its an overlapping entry or disjoint
        // this helps with balancing the dataset accordingly.

        if (entityType.equals(EntityType.NONE)) {
            for (final Word word : words) {
                if (!word.getEntityTag().equals(EntityType.NONE)) {
                    this.overlap = true;
                    break;
                }
            }

            if (!this.overlap) {
                this.disjoint = true;
            }
        }
    }

    public String getEntityType() {
        return entityType;
    }

    public List<Word> getWords() {
        return words;
    }

    public void setPredictedEntityTag(final double probability, final String entityType) {
        this.probability = probability;
        this.predictedEntityTag = entityType;
    }

    public String getPredictedEntityTag() {
        return predictedEntityTag;
    }

    public double getPredictedProbability() {
        return probability;
    }

    /**
     * Check if this n-gram survived the pruning process and should be included in the final output or not
     *
     * @return true or false
     */
    public boolean getSurvivedPruning() {
        return survivedPruning;
    }

    public void setSurvivedPruning(final boolean survivedPruning) {
        this.survivedPruning = survivedPruning;
    }

    public int getStartOffset() {
        return words.get(0).getStartOffset();
    }

    public int getEndOffset() {
        return words.get(words.size() - 1).getEndOffset();
    }

    public int getStartIndex() {
        return startIndex;
    }

    public int getEndIndex() {
        return endIndex;
    }

    public int length() {
        return words.size();
    }

    public String toString() {
        final StringBuilder sb = new StringBuilder();

        sb.append("EntityType=");
        sb.append(entityType);
        sb.append(", String=");

        for (final Word word : words) {
            sb.append(word.getWordString());
            sb.append(" ");
        }

        // Remove final space
        sb.deleteCharAt(sb.length() - 1);

        return sb.toString();
    }

    public boolean isDisjoint() {
        return disjoint;
    }

    public boolean isOverlapping() {
        return overlap;
    }
}
