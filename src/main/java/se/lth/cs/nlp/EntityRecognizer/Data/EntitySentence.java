package se.lth.cs.nlp.EntityRecognizer.Data;

import se.lth.cs.nlp.EntityRecognizer.Corpus.NGram;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Sentence;

public class EntitySentence {
    private final Sentence sentence;
    private final NGram focus;

    public EntitySentence(final Sentence sentence, final NGram focus) {
        this.sentence = sentence;
        this.focus = focus;
    }

    public Sentence getSentence() {
        return sentence;
    }

    public NGram getNGram() {
        return focus;
    }
}
