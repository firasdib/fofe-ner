package se.lth.cs.nlp.EntityRecognizer.Corpus.TAC;

import org.jetbrains.annotations.NotNull;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Corpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.EntityType;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Word;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class TACCorpus extends Corpus {

    public TACCorpus(final String path) {
        final StringJoiner sj = new StringJoiner("\n");

        try (final BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(path), StandardCharsets.UTF_8))) {
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

                final List<String> sentenceWords = new ArrayList<>();

                int nestCount = 0;
                for (final String rawWord : sentence) {
                    if (rawWord.length() > 1) {
                        sentenceWords.add(rawWord);

                        final String[] allMentions = rawWord.split("\t")[3].split(",");

                        nestCount = Math.max(nestCount, allMentions.length);
                    }
                }

                for (int i = 0; i < nestCount; i++) {
                    final List<Word> words = new ArrayList<>();
                    final Set<Integer> mentionIds = new HashSet<>();

                    for (final String wordString : sentenceWords) {
                        final String[] allMentions = wordString.split("\t")[3].split(",");
                        final int mentionIdx = Math.min(i, allMentions.length - 1);
                        final String currentMention = allMentions[mentionIdx];

                        if (currentMention.equals(EntityType.NONE)) {
                            words.add(new TACWord(wordString, -1));

                            continue;
                        }

                        final int mentionId = Integer.parseInt(currentMention.split("#")[1]);

                        if (currentMention.startsWith("B-")) {
                            mentionIds.add(mentionId);
                        }

                        // Some nested mention, ignore for now
                        if (!mentionIds.contains(mentionId)) {
                            words.add(new TACWord(wordString, -1));

                            continue;
                        }

                        words.add(new TACWord(wordString, mentionId));
                    }

                    addSentence(words);
                }
            }
        } catch (final Exception ex) {
            throw new RuntimeException("Unable to parse TAC corpus file", ex);
        }
    }

    @NotNull
    @Override
    protected String getEntityType(final List<Word> words) {
        return Corpus.getIOB2EntityType(words);
    }
}
