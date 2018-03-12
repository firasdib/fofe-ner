package se.lth.cs.nlp.EntityRecognizer.Corpus;

public abstract class Word {
    private String wordString = "";
    private String posTag = "";
    private String chunkTag = "";
    private String entityTag = null;
    private String predictedEntityTag = null;
    private int startOffset;
    private int endOffset;

    public void setEntityTag(final String entityTag) {
        this.entityTag = entityTag;
    }

    public void setWordString(final String wordString) {
        this.wordString = wordString;
    }

    public void setPosTag(final String posTag) {
        this.posTag = posTag;
    }

    public void setChunkTag(final String chunkTag) {
        this.chunkTag = chunkTag;
    }

    public String getWordString() {
        return wordString;
    }

    public String getPosTag() {
        return posTag.toUpperCase();
    }

    public String getChunkTag() {
        return chunkTag.toUpperCase();
    }

    public String getEntityTag() {
        if (entityTag == null) {
            return EntityType.NONE;
        }

        return entityTag;
    }

    public void setPredictedEntityTag(final String predictedEntityTag) {
        this.predictedEntityTag = predictedEntityTag;
    }

    public String getPredictedEntityTag() {
        return predictedEntityTag;
    }

    public String toString() {
        return "String=" + wordString + ", POS=" + posTag + ", Chunk=" + chunkTag + ", Entity=" + entityTag +
                ", PredictedEntity=" + predictedEntityTag;
    }

    @Override
    public boolean equals(final Object o) {
        if (this == o) return true;
        if (!(o instanceof Word)) return false;

        final Word word = (Word) o;

        return entityTag.equals(word.entityTag) &&
                wordString.equals(word.wordString) &&
                posTag.equals(word.posTag) &&
                chunkTag.equals(word.chunkTag);
    }

    @Override
    public int hashCode() {
        return wordString.hashCode();
    }

    public int getStartOffset() {
        return startOffset;
    }

    public void setStartOffset(final int startOffset) {
        this.startOffset = startOffset;
    }

    public int getEndOffset() {
        return endOffset;
    }

    public void setEndOffset(final int endOffset) {
        this.endOffset = endOffset;
    }
}
