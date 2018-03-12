package thrift;

import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TFramedTransport;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransportException;
import se.lth.cs.docforia.Document;
import se.lth.cs.docforia.io.mem.Input;
import se.lth.cs.docforia.memstore.MemoryBinary;

import java.io.IOError;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Created by marcus on 2017-08-17.
 */
public class ThriftLangforiaClient extends LangforiaFrontend.Client implements AutoCloseable {

    private static TProtocol connect(String host, int port) throws TTransportException {
        TFramedTransport transport = new TFramedTransport(new TSocket(host, port));
        transport.open();

        return new TBinaryProtocol(transport);
    }

    public ThriftLangforiaClient(String host, int port) throws TTransportException {
        super(connect(host, port));
    }

    public void annotate(String lang, String pipeline, Document doc) {
        try {
            ByteBuffer byteBuffer = apply_docforia(lang, pipeline, ByteBuffer.wrap(doc.toBytes()));
            Document annodoc = MemoryBinary.decode(new Input(byteBuffer));
            doc.importNodes(annodoc, 0);
        } catch (TException e) {
            throw new IOError(e);
        }
    }

    @Override
    public void close() throws IOException {
        this.iprot_.getTransport().close();
    }
}
