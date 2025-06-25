package org.anjali;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Tokenizer
 *
 * Builds vocabulary from a corpus of .txt/.json files,
 * then encodes/decodes text -> integer token IDs.
 */
public class Tokenizer {
    private final Map<String,Integer> tokenToId = new LinkedHashMap<>();
    private final List<String>        idToToken = new ArrayList<>();
    private static final ObjectMapper M = new ObjectMapper();

    /** Initialize with special tokens */
    public Tokenizer() {
        add("<PAD>");   // padding
        add("<UNK>");   // unknown
        add("<BOS>");   // begin-of-sequence
        add("<EOS>");   // end-of-sequence
    }

    /** Add a token to vocab */
    private void add(String tok) {
        tokenToId.put(tok, idToToken.size());
        idToToken.add(tok);
    }

    /**
     * Scan corpusDir for .txt/.json, count word frequencies,
     * and add all words occurring ≥ minCount to vocab.
     */
    public void buildVocab(Path corpusDir, int minCount) throws IOException {
        Map<String,Integer> freq = new HashMap<>();

        // walk all files
        try (var stream = Files.walk(corpusDir)) {
            for (Path f : stream.filter(Files::isRegularFile).collect(Collectors.toList())) {
                String blob;
                if (f.toString().endsWith(".json")) {
                    // read JSON array under root or top level
                    JsonNode root = M.readTree(f.toFile());
                    JsonNode arr  = root.has("root") ? root.get("root") : root;
                    StringBuilder sb = new StringBuilder();
                    for (JsonNode item : arr) {
                        if (item.has("text")) {
                            sb.append(item.get("text").asText()).append(" ");
                        }
                    }
                    blob = sb.toString();
                } else if (f.toString().endsWith(".txt")) {
                    // read plain text
                    blob = Files.readString(f);
                } else {
                    continue;
                }
                // split on non-word characters
                for (String w : blob.toLowerCase().split("\\W+")) {
                    if (!w.isBlank()) {
                        freq.merge(w, 1, Integer::sum);
                    }
                }
            }
        }

        // add all tokens above threshold, sorted alphabetically
        freq.entrySet().stream()
            .filter(e -> e.getValue() >= minCount)
            .map(Map.Entry::getKey)
            .sorted()
            .forEach(this::add);

        System.out.printf("Built vocab: %d tokens (%d special)%n",
            idToToken.size(), 4);
    }

    /**
     * Encode for training: [BOS] + tokens + [EOS]
     * unknown words -> <UNK>
     */
    public int[] encode(String text) {
        List<Integer> ids = new ArrayList<>();
        ids.add(tokenToId.get("<BOS>"));
        for (String w : text.toLowerCase().split("\\W+")) {
            if (!w.isBlank()) {
                ids.add(tokenToId.getOrDefault(w, tokenToId.get("<UNK>")));
            }
        }
        ids.add(tokenToId.get("<EOS>"));
        return ids.stream().mapToInt(i -> i).toArray();
    }

    /**
     * Encode prompt for generation: no EOS so we can loop until generated EOS.
     */
    public int[] encodePrompt(String text) {
        List<Integer> ids = new ArrayList<>();
        ids.add(tokenToId.get("<BOS>"));
        for (String w : text.toLowerCase().split("\\W+")) {
            if (!w.isBlank()) {
                ids.add(tokenToId.getOrDefault(w, tokenToId.get("<UNK>")));
            }
        }
        return ids.stream().mapToInt(i -> i).toArray();
    }

    /**
     * Decode a full token ID list (includes generated EOS).
     * Drops special BOS/EOS tokens.
     */
    public String decodeAll(List<Integer> ids) {
        List<String> out = new ArrayList<>();
        for (int id : ids) {
            String t = idToToken.get(id);
            if (t.equals("<BOS>") || t.equals("<EOS>")) continue;
            out.add(t);
        }
        return String.join(" ", out);
    }

    // getters for integration
    public int getEosId()     { return tokenToId.get("<EOS>"); }
    public int getVocabSize() { return idToToken.size(); }
}
