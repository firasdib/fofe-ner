package se.lth.cs.nlp.EntityRecognizer.Corpus;

import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.NetworkMode;

import java.util.Arrays;
import java.util.List;

public class EntityType {
    public static final String MISC = "MISC";
    public static final String LOC = "LOC";
    public static final String ORG = "ORG";
    public static final String PER = "PER";
    public static final String GPE_NAM = "GPE_NAM";
    public static final String GPE_NOM = "GPE_NOM";
    public static final String FAC_NAM = "FAC_NAM";
    public static final String FAC_NOM = "FAC_NOM";
    public static final String PER_NAM = "PER_NAM";
    public static final String PER_NOM = "PER_NOM";
    public static final String LOC_NAM = "LOC_NAM";
    public static final String LOC_NOM = "LOC_NOM";
    public static final String ORG_NAM = "ORG_NAM";
    public static final String ORG_NOM = "ORG_NOM";
    public static final String TTL_NAM = "TTL_NAM"; // TTL_NAM = PER_NOM
    public static final String NONE = "O";

    // SUC specific
    public static final String PERSON = "person";
    public static final String ANIMAL = "animal";
    public static final String MYTH = "myth";
    public static final String PLACE = "place";
    public static final String INST = "inst";
    public static final String PRODUCT = "product";
    public static final String WORK = "work";
    public static final String EVENT = "event";
    public static final String OTHER = "other";

    public static List<String> getEntityTypes(final NetworkMode networkMode) {
        switch (networkMode) {
            case CONLL2002:
            case CONLL2003:
            case DOCFORIA_CONLL:
            case SUC:
            case DOCFORIA_SUC:
                return Arrays.asList(MISC, LOC, ORG, PER, NONE);
            case DOCFORIA_TAC:
            case TAC:
                return Arrays.asList(LOC_NAM, LOC_NOM, ORG_NAM, ORG_NOM, PER_NAM, PER_NOM, GPE_NAM, GPE_NOM, FAC_NAM, FAC_NOM, NONE);
            case SUC_RAW:
                return Arrays.asList(PERSON, ANIMAL, MYTH, PLACE, INST, PRODUCT, WORK, EVENT, OTHER, NONE);
            default:
                throw new IllegalArgumentException("Unknown network mode '" + networkMode + "'");
        }
    }
}
