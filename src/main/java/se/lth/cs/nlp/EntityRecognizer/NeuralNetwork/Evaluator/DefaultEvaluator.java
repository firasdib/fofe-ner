package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Evaluator;

import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import se.lth.cs.nlp.EntityRecognizer.Data.NGramDataSet;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.NNUtils;

public class DefaultEvaluator extends Evaluator {
    private static final Logger log = LoggerFactory.getLogger(DefaultEvaluator.class);

    private Evaluation eval;
    private int count;

    @Override
    protected void eval(final INDArray modelOutput, final NGramDataSet dataSet) {
        if (count++ % 100 == 0) {
            log.info("Evaluating batch {}/{}...", count, batchCount);
        }

        // There should only ever be one output
        eval.eval(dataSet.getLabels()[0], modelOutput);
    }

    @Override
    protected void preEval() {
        log.info("Evaluating model...");

        eval = new Evaluation(metaData.getEntityTypes());
        count = 0;
    }

    @Override
    protected void postEval(final long startTime) {
        log.info(eval.stats());

        log.info("Evaluation done! Took {}", NNUtils.formatTime(System.currentTimeMillis() - startTime));
    }

    public Evaluation getEval() {
        return eval;
    }
}
