package se.lth.cs.nlp.EntityRecognizer.Corpus;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.function.Function;

public abstract class CoNLLCorpus extends Corpus {
    protected void parse(final String conllPath, final Function<String, Word> string2Word) {
        final StringJoiner sj = new StringJoiner("\n");

        try (final BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(conllPath), StandardCharsets.UTF_8))) {
            String line;

            while ((line = br.readLine()) != null) {
                sj.add(line);
            }

            final String document = sj.toString();
            final String[] sentences = document.split("\\n\\n");

            for (final String rawSentence : sentences) {
                final String[] sentence = rawSentence.trim().split("\\n");

                if (sentence.length == 0 || rawSentence.trim().toLowerCase().startsWith("-docstart-")) {
                    continue;
                }

                final List<Word> wordsInSentence = new ArrayList<>();

                for (final String rawWord : sentence) {
                    if (rawWord.length() > 1) {
                        final Word word = string2Word.apply(rawWord);

                        wordsInSentence.add(word);
                    }
                }

                this.addSentence(wordsInSentence);
            }
        } catch (final Exception ex) {
            throw new RuntimeException("Unable to parse CoNLL corpus file", ex);
        }
    }
}
