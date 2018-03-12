package se.lth.cs.nlp.EntityRecognizer.Corpus.CoNLL2003;

import se.lth.cs.nlp.EntityRecognizer.Corpus.Word;

public class CoNLL2003Word extends Word {
    public CoNLL2003Word(final String conllRow) {
        // The first item on each line is a wordString, the second a part-of-speech (POS) tag,
        // the third a syntactic chunk tag and the fourth the named entity tag.
        // The chunk tags and the named entity tags have the format I-TYPE which means that
        // the wordString is inside a phrase of type TYPE.

        final String[] parts = conllRow.split(" ");

        this.setWordString(parts[0]);
        this.setPosTag(parts[1]);
        this.setChunkTag(parts[2]);
        this.setEntityTag(parts[3]);
    }
}
