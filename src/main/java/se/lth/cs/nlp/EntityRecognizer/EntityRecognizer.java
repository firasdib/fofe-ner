package se.lth.cs.nlp.EntityRecognizer;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import se.lth.cs.nlp.EntityRecognizer.CLI.EvalCommand;
import se.lth.cs.nlp.EntityRecognizer.CLI.ServerCommand;
import se.lth.cs.nlp.EntityRecognizer.CLI.TrainCommand;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.NNManager;
import se.lth.cs.nlp.EntityRecognizer.Server.Server;

public class EntityRecognizer {
    private static final Logger log = LoggerFactory.getLogger(EntityRecognizer.class);

    public static void main(final String[] args) {
        // Register a Default Uncaught Exception Handler
        Thread.setDefaultUncaughtExceptionHandler((t, e) -> log.error("Uncaught exception", e));

        new ProgramSwitch(args);
    }

    @SuppressWarnings("CanBeFinal")
    private static class ProgramSwitch {
        @Parameter(names = {"-h", "--help"}, description = "Shows usage information and shuts down the application", help = true)
        private boolean help = false;

        private static final String SERVER = "server";
        private static final String TRAIN = "train";
        private static final String EVAL = "eval";

        public ProgramSwitch(final String[] args) {
            final TrainCommand tc = new TrainCommand();
            final EvalCommand ec = new EvalCommand();
            final ServerCommand sc = new ServerCommand();

            final JCommander jCommander = JCommander.newBuilder()
                    .programName("FOFE NER")
                    .addObject(this)
                    .addCommand(SERVER, sc)
                    .addCommand(EVAL, ec)
                    .addCommand(TRAIN, tc)
                    .build();

            jCommander.parse(args);

            if (help || jCommander.getParsedCommand() == null) {
                jCommander.usage();

                System.exit(0);

                return;
            }

            // Adjust GC settings for perf
            Nd4j.create(1); // Need to do this first
            Nd4j.getMemoryManager().setAutoGcWindow(5000);

            switch (jCommander.getParsedCommand().toLowerCase()) {
                case SERVER:
                    Config.readFile(sc.configFile);

                    Server.start();
                    break;
                case EVAL:
                    Config.readFile(ec.configFile);

                    NNManager.load(ec);
                    break;
                case TRAIN:
                    Config.readFile(tc.configFile);

                    NNManager.create(tc);
                    break;
                default:
                    throw new RuntimeException("What?");
            }
        }
    }
}
