package se.lth.cs.nlp.EntityRecognizer.Server;

import org.apache.thrift.server.THsHaServer;
import org.apache.thrift.server.TServer;
import org.apache.thrift.transport.TFramedTransport;
import org.apache.thrift.transport.TNonblockingServerSocket;
import org.apache.thrift.transport.TNonblockingServerTransport;
import org.apache.thrift.transport.TTransportException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import thrift.LangforiaFrontend;

import java.net.InetSocketAddress;

public class Server {
    private static final Logger log = LoggerFactory.getLogger(Server.class);

    public static final String DEFAULT_CONFIG_FILE = "./server.json";

    public static void start() {
        try {
            final int port = Config.getInt("port");

            final LangforiaFrontend.Processor<Frontend> processor = new LangforiaFrontend.Processor<>(new Frontend());

            final TNonblockingServerTransport serverTransport = new TNonblockingServerSocket(
                    new InetSocketAddress("0.0.0.0", port));

            final TServer server = new THsHaServer(new THsHaServer.Args(serverTransport).transportFactory(
                    new TFramedTransport.Factory(128 * 1024 * 1024)).processor(processor));

            log.info("Starter server at port {}", port);

            server.serve();
        } catch (final TTransportException e) {
            log.error("Transport Exception", e);
        }
    }
}
