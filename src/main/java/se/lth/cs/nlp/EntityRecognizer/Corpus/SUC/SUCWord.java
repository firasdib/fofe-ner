package se.lth.cs.nlp.EntityRecognizer.Corpus.SUC;

import se.lth.cs.nlp.EntityRecognizer.Corpus.EntityType;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Word;

public class SUCWord extends Word {
    public SUCWord(final String rawWord, final boolean useRawClasses) {
        final String[] parts = rawWord.split("\\t+");

        setWordString(parts[1]);

        final String entityTag;

        // Turns out this corpus is inconsistent....
        // What we do know is that the final 3 columns seem to be consistent, so we use those...

        final int len = parts.length;
        final String entityIOB = parts[len - 3];
        final String entityType = parts[len - 2];

        if (entityIOB.equals("O")) {
            entityTag = EntityType.NONE;
        } else {
            entityTag = useRawClasses
                    ? entityIOB + '-' + entityType
                    : this.transformEntity(entityIOB, entityType);
        }

        setEntityTag(entityTag);
    }

    private String transformEntity(final String entityIOB, final String entityType) {
        // Transformation in accordance to Andreas Salomonssons paper
        final String newEntityType;

        switch (entityType.toLowerCase()) {
            case "person":
                newEntityType = "PER";
                break;
            case "place":
                newEntityType = "LOC";
                break;
            case "inst":
                newEntityType = "ORG";
                break;
            default:
                newEntityType = "MISC";
        }

        return entityIOB + "-" + newEntityType;
    }
}
