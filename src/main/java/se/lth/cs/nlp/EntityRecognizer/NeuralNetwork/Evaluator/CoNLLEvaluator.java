package se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.Evaluator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.zeroturnaround.exec.ProcessExecutor;
import se.lth.cs.nlp.EntityRecognizer.Corpus.*;

import java.io.*;
import java.util.concurrent.TimeoutException;

public class CoNLLEvaluator extends DefaultEvaluator {
    private static final Logger log = LoggerFactory.getLogger(CoNLLEvaluator.class);

    public static final String CONLL_OUTPUT = "./conll.output";

    @Override
    protected void postEval(final long startTime) {
        log.info("Writing CoNLL output file...");

        final File output = new File(CONLL_OUTPUT);

        try (final BufferedWriter bw = new BufferedWriter(new FileWriter(output))) {
            for (final Sentence sentence : metaData.getCorpus().getSentences()) {
                for (final Word word : sentence.getWords()) {
                    final StringBuilder sb = new StringBuilder();

                    sb.append(word.getWordString());
                    sb.append(" ");

                    if (!word.getPosTag().equals("")) {
                        sb.append(word.getPosTag());
                        sb.append(" ");
                    }

                    if (!word.getChunkTag().equals("")) {
                        sb.append(word.getChunkTag());
                        sb.append(" ");
                    }

                    // Remove IOB-encoding since the system doesn't output IOB specific encoding
                    final String entityTag = word.getEntityTag().replaceFirst("[IB]-", "");

                    sb.append(entityTag);
                    sb.append(" ");
                    sb.append(word.getPredictedEntityTag());

                    bw.write(sb.toString());
                    bw.write("\n");
                }

                bw.write("\n");
            }
        } catch (final IOException e) {
            log.error("Unable to write to file {}", output.getAbsolutePath(), e);
        }

        super.postEval(startTime);

        try (final InputStream fis = new FileInputStream(output)) {
            final String conllEvalScript = new ProcessExecutor().command("./conlleval.pl", "-r")
                    .redirectInput(fis)
                    .readOutput(true).execute()
                    .outputUTF8();

            log.info("\n" + conllEvalScript);
        } catch (IOException | InterruptedException | TimeoutException e) {
            log.error("Unable to run conll evaluation script", e);
        }

    }
}
