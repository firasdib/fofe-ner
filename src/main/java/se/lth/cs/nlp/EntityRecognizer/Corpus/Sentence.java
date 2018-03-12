package se.lth.cs.nlp.EntityRecognizer.Corpus;

import java.util.List;
import java.util.stream.Collectors;

public class Sentence {
    private final List<Word> words;
    private final List<NGram> nGrams;

    public Sentence(final List<Word> words, final List<NGram> nGrams) {
        this.words = words;
        this.nGrams = nGrams;
    }

    public List<Word> getWords() {
        return words;
    }

    public List<NGram> getAllNGrams() {
        return nGrams;
    }

    public String toString() {
        return words.stream()
                .map(Word::getWordString)
                .collect(Collectors.joining(" "));
    }
}
