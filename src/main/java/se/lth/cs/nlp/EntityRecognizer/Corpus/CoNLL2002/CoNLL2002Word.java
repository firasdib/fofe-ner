package se.lth.cs.nlp.EntityRecognizer.Corpus.CoNLL2002;

import se.lth.cs.nlp.EntityRecognizer.Corpus.Word;

public class CoNLL2002Word extends Word {
    public CoNLL2002Word(final String conllRow) {
        final String[] parts = conllRow.split(" ");

        this.setWordString(parts[0]);
        this.setEntityTag(parts[1]);
    }
}
