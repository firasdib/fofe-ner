package se.lth.cs.nlp.EntityRecognizer.Server;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import org.apache.thrift.TException;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import se.lth.cs.docforia.graph.text.Sentence;
import se.lth.cs.docforia.graph.text.Token;
import se.lth.cs.docforia.io.mem.Input;
import se.lth.cs.docforia.memstore.MemoryBinary;
import se.lth.cs.docforia.memstore.MemoryDocument;
import se.lth.cs.nlp.EntityRecognizer.Config.Config;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Corpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Docforia.DocforiaCorpus;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Docforia.DocforiaWord;
import se.lth.cs.nlp.EntityRecognizer.Corpus.Word;
import se.lth.cs.nlp.EntityRecognizer.NeuralNetwork.NeuralNetwork;
import se.lth.cs.nlp.tokenization.Segment;
import thrift.LangforiaFrontend;

import java.nio.ByteBuffer;
import java.util.*;
import java.util.stream.Collectors;

public class Frontend implements LangforiaFrontend.Iface {
    private static final Logger log = LoggerFactory.getLogger(Frontend.class);

    private final Map<String, List<Pipeline>> configurations = new HashMap<>();
    private final Map<String, Pipeline> lang2pipeline = new HashMap<>();
    private StanfordCoreNLP coreNLP;

    private final Map<String, MemoryDocument> marcusCache = new HashMap<>();

    private String currentConfig = null;

    public Frontend() {
        final List<JSONObject> languages = Config.getList("languages");

        for (final JSONObject languageObj : languages) {
            final String language = (String) languageObj.get("language");

            @SuppressWarnings("unchecked") final List<JSONObject> pipelineObjects = (JSONArray) languageObj.get("pipelines");
            final List<Pipeline> pipelines = new ArrayList<>();

            for (final JSONObject pipelineObj : pipelineObjects) {
                final String name = (String) pipelineObj.get("name");
                final String model = (String) pipelineObj.get("model");
                final String mode = (String) pipelineObj.get("mode");
                final String config = (String) pipelineObj.get("config");

                if (name == null) {
                    throw new RuntimeException("Field 'name' is not set for pipeline " + language);
                }

                if (model == null) {
                    throw new RuntimeException("Field 'model' is not set for pipeline " + language);
                }

                if (mode == null) {
                    throw new RuntimeException("Field 'mode' is not set for pipeline " + language);
                }

                if (config == null) {
                    throw new RuntimeException("Field 'config' is not set for pipeline " + language);
                }

                final String key = language + name;
                final Pipeline pipeline = new Pipeline(model, mode, config, language, name);

                pipelines.add(pipeline);

                lang2pipeline.put(key, pipeline);

                log.info("+ Created pipeline for language '{}': {}", language, name);
            }

            configurations.put(language, pipelines);
        }

        log.info("Loading CoreNLP");

        // See https://github.com/stanfordnlp/CoreNLP/blob/master/src/edu/stanford/nlp/pipeline/StanfordCoreNLP-chinese.properties
        final Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit");
        props.setProperty("customAnnotatorClass.segment", "edu.stanford.nlp.pipeline.ChineseSegmenterAnnotator");
        props.setProperty("segment.model", "edu/stanford/nlp/models/segmenter/chinese/ctb.gz");
        props.setProperty("segment.sighanCorporaDict", "edu/stanford/nlp/models/segmenter/chinese");
        props.setProperty("segment.serDictionary", "edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz");
        props.setProperty("segment.sighanPostProcessing", "true");
        props.setProperty("tokenize.language", "zh");
        props.setProperty("ssplit.boundaryTokenRegex", "[.。]|[!?！？]+");

        coreNLP = new StanfordCoreNLP(props);
    }

    @Override
    public void ping() throws TException {
        System.out.println("Pong");
    }

    @Override
    public List<String> languages() throws TException {
        return new ArrayList<>(configurations.keySet());
    }

    @Override
    public List<String> pipelines(final String lang) throws TException {
        return configurations.get(lang).stream()
                .map(Pipeline::getName)
                .collect(Collectors.toList());
    }

    @Override
    public ByteBuffer apply_docforia(final String langString, final String pipelineString, final ByteBuffer doc) throws TException {
        return ByteBuffer.wrap(evaluate(langString, pipelineString, doc).toBytes());
    }

    private synchronized MemoryDocument evaluate(final String langString, final String pipelineString, final ByteBuffer doc) {
        final MemoryDocument document = MemoryBinary.decode(new Input(doc));

        final String text = document.text();
        final String cacheStr = langString + pipelineString + text;

        if (marcusCache.get(cacheStr) != null) {
            return marcusCache.get(cacheStr);
        }

        log.info("Loading pipeline for {} ({})", langString, pipelineString);
        final Pipeline pipeline = lang2pipeline.get(langString + pipelineString);

        // Only reload config if its different from the currently loaded config
        if (!pipeline.getConfig().equals(currentConfig)) {
            currentConfig = pipeline.getConfig();

            Config.readFile(currentConfig);
        }

        // Must be read after config has been loaded
        final boolean chineseMode = Config.getBool("chineseMode");

        final NeuralNetwork network = pipeline.getNetwork();

        // Create a Docforia specific corpus
        final DocforiaCorpus corpus = (DocforiaCorpus) Corpus.createCorpus(network.getNetworkMode());
        corpus.setDocument(document);

        if (chineseMode) {
            final Annotation stanfordDocument = new Annotation(text);
            coreNLP.annotate(stanfordDocument);

            final List<CoreMap> sentences = stanfordDocument.get(CoreAnnotations.SentencesAnnotation.class);
            for (final CoreMap sentence : sentences) {
                final List<Word> words = new ArrayList<>();

                for (final CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                    final int startPos = token.get(CoreAnnotations.CharacterOffsetBeginAnnotation.class);
                    final int endPos = token.get(CoreAnnotations.CharacterOffsetEndAnnotation.class);

                    final Token wordToken = new Token(document).setRange(startPos, endPos);
                    final DocforiaWord word = new DocforiaWord(wordToken);

                    words.add(word);
                }

                if (words.size() > 0) {
                    corpus.addSentence(words);

                    new Sentence(document).setRange(words.get(0).getStartOffset(),
                            words.get(words.size() - 1).getEndOffset());
                }
            }
        } else {
            final List<Segment> segments = Segment.segment(langString, text);

            for (final Segment segment : segments) {
                se.lth.cs.nlp.tokenization.Token firstToken = segment.get(0);
                se.lth.cs.nlp.tokenization.Token lastToken = segment.get(segment.size() - 1);

                new Sentence(document).setRange(firstToken.start, lastToken.end);

                final List<Word> words = new ArrayList<>();
                for (final se.lth.cs.nlp.tokenization.Token token : segment) {
                    final Token wordToken = new Token(document).setRange(token.start, token.end);

                    words.add(new DocforiaWord(wordToken));
                }

                corpus.addSentence(words);
            }
        }

        network.evaluate(corpus);

        marcusCache.put(cacheStr, document);

        return document;
    }
}
