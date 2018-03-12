package se.lth.cs.nlp.EntityRecognizer.Corpus.CoNLL2003;

import org.jetbrains.annotations.NotNull;
import se.lth.cs.nlp.EntityRecognizer.Corpus.CoNLLCorpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Corpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Word;

import java.util.List;

public class CoNLL2003Corpus extends CoNLLCorpus {
    public CoNLL2003Corpus(final String conllPath) {
        parse(conllPath, CoNLL2003Word::new);
    }

    @NotNull
    @Override
    protected String getEntityType(List<Word> words) {
        return Corpus.getIOBEntityType(words);
    }
}
