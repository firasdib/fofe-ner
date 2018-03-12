package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Evaluator;

import org.nd4j.linalg.api.ndarray.INDArray;
import se.lth.cs.docforia.Document;
import se.lth.cs.docforia.graph.TokenProperties;
import se.lth.cs.docforia.graph.text.NamedEntity;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Docforia.DocforiaCorpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Docforia.DocforiaWord;
import se.lth.cs.nlp.EntityRecognizer.Corpus.NGram;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Sentence;
import se.lth.cs.nlp.EntityRecognizer.Data.NGramDataSet;

public class DocforiaEvaluator extends Evaluator {
    @Override
    protected void eval(final INDArray modelOutput, final NGramDataSet dataSet) {

    }

    @Override
    protected void preEval() {

    }

    @Override
    protected void postEval(long startTime) {
        // Annotation is now finished, including all pruning, and we can copy this over to docforia
        final Document document = ((DocforiaCorpus) metaData.getCorpus()).getDocument();

        for (final Sentence sentence : metaData.getCorpus().getSentences()) {
            for (final NGram nGram : sentence.getAllNGrams()) {
                if (nGram.getSurvivedPruning()) {
                    nGram.getWords().forEach(word ->
                            ((DocforiaWord) word).getToken()
                                    .putProperty(TokenProperties.NE, nGram.getPredictedEntityTag())
                                    .putProperty("confidence", nGram.getPredictedProbability()));

                    final NamedEntity namedEntity = new NamedEntity(document)
                            .setRange(nGram.getStartOffset(), nGram.getEndOffset())
                            .putProperty("confidence", nGram.getPredictedProbability());

                    // If this n-gram is concatenated with another word, split it out
                    if (nGram.getPredictedEntityTag().contains("_")) {
                        final String[] split = nGram.getPredictedEntityTag().split("_");

                        namedEntity.setLabel(split[0])
                                .putProperty("type", split[1]);
                    } else {
                        namedEntity.setLabel(nGram.getPredictedEntityTag());
                    }
                }
            }
        }
    }
}
