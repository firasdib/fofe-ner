package se.lth.cs.nlp.EntityRecognizer.Corpus.TAC;

import se.lth.cs.nlp.EntityRecognizer.Corpus.EntityType;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Word;

public class TACWord extends Word {
    public TACWord(final String rawWord, final int mentionId) {
        final String[] parts = rawWord.split("\\t+");

        // paths[0] is the original text, parts[1] is the mention-indexed one
        setWordString(parts[1]);

        if (mentionId == -1) {
            setEntityTag(EntityType.NONE);
        } else {
            // Looks something like B-GPE#3231, we want B-GPE
            final String[] entityTags = parts[3].split(",");
            final String[] nomOrNam = parts[4].split(",");

            for (int i = 0; i < entityTags.length; i++) {
                final String[] split = entityTags[i].split("#");

                final String entityTag = split[0];
                final int entityId = Integer.parseInt(split[1]);

                // Only add the label for the selected mention id
                if (entityId != mentionId) {
                    continue;
                }

                final String extra = nomOrNam[i].split("#")[0].substring(2);
                final String newEntityTag = entityTag + "_" + extra;

                if (newEntityTag.contains(EntityType.TTL_NAM)) {
                    setEntityTag(newEntityTag.replace(EntityType.TTL_NAM, EntityType.PER_NOM));
                } else {
                    setEntityTag(entityTag + "_" + extra);
                }
            }
        }
    }
}
