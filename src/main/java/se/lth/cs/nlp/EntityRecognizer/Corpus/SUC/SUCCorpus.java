package se.lth.cs.nlp.EntityRecognizer.Corpus.SUC;

import org.jetbrains.annotations.NotNull;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Corpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Word;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class SUCCorpus extends Corpus {

    public SUCCorpus(final String corpusFilePath, final boolean rawClasses) {
        final StringJoiner sj = new StringJoiner("\n");

        try (final BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(corpusFilePath), StandardCharsets.UTF_8))) {
            String line;

            while ((line = br.readLine()) != null) {
                sj.add(line);
            }

            final String document = sj.toString();
            final String[] sentences = document.split("\\n\\n");

            for (final String rawSentence : sentences) {
                final String[] sentence = rawSentence.trim().split("\\n");

                if (sentence.length == 0) {
                    continue;
                }

                final List<Word> sentenceWords = new ArrayList<>();

                for (final String rawWord : sentence) {
                    if (rawWord.length() > 1) {
                        sentenceWords.add(new SUCWord(rawWord, rawClasses));
                    }
                }

                addSentence(sentenceWords);
            }
        } catch (final Exception ex) {
            throw new RuntimeException("Unable to parse SUC corpus file", ex);
        }
    }

    public SUCCorpus(final String corpusFilePath) {
        this(corpusFilePath, false);
    }

    @NotNull
    @Override
    protected String getEntityType(final List<Word> words) {
        return Corpus.getIOB2EntityType(words);
    }
}
