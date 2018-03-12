package se.lth.cs.nlp.EntityRecognizer.Corpus.Docforia;

import se.lth.cs.docforia.graph.text.Token;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Word;

public class DocforiaWord extends Word {
    private final Token token;

    public DocforiaWord(final Token token) {
        this.token = token;

        setWordString(token.text());

        setStartOffset(token.getStart());
        setEndOffset(token.getEnd());
    }

    public Token getToken() {
        return token;
    }
}
