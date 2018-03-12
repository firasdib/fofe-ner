package se.lth.cs.nlp.EntityRecognizer.CLI;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import se.lth.cs.nlp.EntityRecognizer.Server.Server;

@SuppressWarnings("CanBeFinal")
@Parameters(commandDescription = "Start the server")
public class ServerCommand {
    @Parameter(names = {"-c", "--config"}, description = "Specify which configuration file to use")
    public String configFile = Server.DEFAULT_CONFIG_FILE;
}
