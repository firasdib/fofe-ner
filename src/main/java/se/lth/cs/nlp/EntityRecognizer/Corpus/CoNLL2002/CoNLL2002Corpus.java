package se.lth.cs.nlp.EntityRecognizer.Corpus.CoNLL2002;

import org.jetbrains.annotations.NotNull;
import se.lth.cs.nlp.EntityRecognizer.Corpus.CoNLLCorpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Corpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Word;

import java.util.List;

public class CoNLL2002Corpus extends CoNLLCorpus {
    public CoNLL2002Corpus(final String conllPath) {
        parse(conllPath, CoNLL2002Word::new);
    }

    @NotNull
    @Override
    protected String getEntityType(final List<Word> words) {
        return Corpus.getIOB2EntityType(words);
    }
}
