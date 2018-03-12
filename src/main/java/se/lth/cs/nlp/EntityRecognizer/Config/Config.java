package se.lth.cs.nlp.EntityRecognizer.Config;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Config {
    private static final Logger log = LoggerFactory.getLogger(Config.class);

    private static final List<JSONObject> configs = new ArrayList<>();

    public static void readFile(final String... files) {
        configs.clear();

        final JSONParser parser = new JSONParser();

        try {
            for (final String file : files) {
                configs.add((JSONObject) parser.parse(new FileReader(file)));
            }
        } catch (Exception e) {
            log.error("Unable to read config file", e);
        }

        final int maxLength = maxLength();

        log.info("=================================================");
        log.info("The following configuration parameters are being used:");
        for (final JSONObject config : configs) {
            final Object[] keys = config.keySet().toArray();
            Arrays.sort(keys);

            for (final Object key : keys) {
                log.info("{}{}: {}",
                        String.join("", Collections.nCopies(maxLength - ((String) key).length(), " ")),
                        key,
                        config.get(key));
            }
        }
        log.info("=================================================");
    }

    private static int maxLength() {
        int maxLength = 0;

        for (final JSONObject config : configs) {
            for (final Object key : config.keySet()) {
                maxLength = Math.max(maxLength, ((String) key).length());
            }
        }

        return maxLength;
    }

    public static void setAttr(final String key, final String value) {
        for (final JSONObject config : configs) {
            for (final Object ignored : config.keySet()) {
                //noinspection unchecked
                config.put(key, value);
            }
        }
    }

    public static String getString(final String key) {
        return (String) getValue(key);
    }

    public static int getInt(final String key) {
        return Integer.parseInt(getString(key), 10);
    }

    public static double getDouble(final String key) {
        return Double.parseDouble(getString(key));
    }

    public static boolean getBool(final String key) {
        return Boolean.parseBoolean(getString(key));
    }

    @SuppressWarnings("unchecked")
    public static List<JSONObject> getList(final String key) {
        final List<JSONObject> list = new ArrayList<>();
        final JSONArray arr = (JSONArray) getValue(key);

        if (arr == null) {
            return list;
        }

        list.addAll(arr);

        return list;
    }

    private static Object getValue(final String key) {
        if (configs.size() == 0) {
            throw new IllegalStateException("Config not initialized properly");
        }

        for (final JSONObject config : configs) {
            if (config.containsKey(key)) {
                return config.get(key);
            }
        }

        throw new IllegalArgumentException("The key '" + key + "' does not exist in the configuration file(s)");
    }
}
