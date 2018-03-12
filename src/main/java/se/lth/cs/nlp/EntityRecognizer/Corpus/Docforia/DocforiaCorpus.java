package se.lth.cs.nlp.EntityRecognizer.Corpus.Docforia;

import org.jetbrains.annotations.NotNull;
import se.lth.cs.docforia.Document;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Corpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.EntityType;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Word;

import java.util.List;

public class DocforiaCorpus extends Corpus {
    private Document document;

    public void setDocument(final Document document) {
        this.document = document;
    }

    public Document getDocument() {
        return document;
    }

    @NotNull
    @Override
    protected String getEntityType(List<Word> words) {
        // This is not used in the docforia case (we're not training, just evaluating)
        return EntityType.NONE;
    }
}
