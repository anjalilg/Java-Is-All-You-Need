package org.anjali;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import me.tongfei.progressbar.ProgressBar;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.*;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * CorpusTrainer
 *
 * Reads all .txt and .json files from a specified corpus directory,
 * tokenizes each text, and trains a given LanguageModel via
 * simple autoregressive negative-log-likelihood gradient updates.
 *
 * Displays a console progress bar and logs per-epoch loss.
 */
public class CorpusTrainer {
    private static final Logger logger   = LoggerFactory.getLogger(CorpusTrainer.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    /**
     * Train the language model.
     *
     * @param lm        the LanguageModel to train
     * @param corpusDir directory containing .txt/.json files
     * @param lr        learning rate
     * @param epochs    number of epochs
     */
    public static void train(LanguageModel lm, Path corpusDir,
                             double lr, int epochs) throws IOException {
        Tokenizer tok = lm.getTokenizer();

        // List all files in corpusDir
        List<Path> files = Files.list(corpusDir).toList();
        logger.info("Found {} files to train on in {}", files.size(), corpusDir);

        // Count total training examples ahead of time
        int totalExamples = countExamples(tok, files);
        logger.info("Total training examples: {}", totalExamples);

        // Loop over epochs
        for (int epoch = 1; epoch <= epochs; epoch++) {
            logger.info("Starting epoch {}/{}", epoch, epochs);
            double totalLoss = 0;
            int processed = 0;

            // ProgressBar: title + totalExamples
            try (ProgressBar pb = new ProgressBar("Epoch " + epoch, totalExamples)) {
                for (Path f : files) {
                    String fname = f.getFileName().toString();
                    logger.info(" Processing file {}", fname);

                    // Read text from .txt or .json
                    if (fname.endsWith(".txt")) {
                        String text = Files.readString(f);
                        var res = processText(tok, lm, text, lr, pb);
                        totalLoss += res.loss;
                        processed += res.count;

                    } else if (fname.endsWith(".json")) {
                        JsonNode root = MAPPER.readTree(f.toFile());
                        JsonNode arr  = root.has("root") ? root.get("root") : root;
                        Iterator<JsonNode> it = arr.elements();
                        while (it.hasNext()) {
                            JsonNode node = it.next();
                            if (node.has("text")) {
                                var res = processText(tok, lm,
                                        node.get("text").asText(), lr, pb);
                                totalLoss += res.loss;
                                processed += res.count;
                            }
                        }
                    }
                }
            }

            // Log average loss for this epoch
            double avgLoss = totalLoss / Math.max(1, processed);
            logger.info("Epoch {} complete ▶ avg loss={}", epoch, avgLoss);
        }
    }

    /**
     * Process one piece of text: encode, forward, compute loss,
     * backprop through head + embedding, then step progress bar.
     */
    private static Result processText(Tokenizer tok,
                                      LanguageModel lm,
                                      String text,
                                      double lr,
                                      ProgressBar pb) {
        int[] tokens = tok.encode(text);
        double loss = 0;
        int count = 0;

        // For each next-token prediction
        for (int i = 1; i < tokens.length; i++) {
            int target = tokens[i];
            int[] seq = Arrays.copyOfRange(tokens, 0, i);

            // 1) Forward pass through embedding + transformer stack
            double[][] hidden = lm.getEmbeddingLayer().forward(seq);
            for (var layer : lm.getLayers()) {
                hidden = layer.forward(hidden);
            }
            double[] logits = lm.getOutputHead()
                                .forward(hidden)[hidden.length - 1];

            // 2) Compute negative log-likelihood loss
            double[] probs = softmax(logits);
            loss += -Math.log(probs[target] + 1e-12);

            // 3) Gradient on output head
            double[] gradLogits = new double[probs.length];
            for (int j = 0; j < probs.length; j++) {
                gradLogits[j] = probs[j] - (j == target ? 1.0 : 0.0);
            }
            lm.getOutputHead().backpropLastToken(hidden[hidden.length - 1],
                                                 gradLogits, lr);

            // 4) Backpropagate into embedding
            double[][] W = lm.getOutputHead().getProj().getWeights();
            double[] gradE = new double[hidden[0].length];
            for (int k = 0; k < gradE.length; k++) {
                double sum = 0;
                for (int j = 0; j < gradLogits.length; j++) {
                    sum += gradLogits[j] * W[k][j];
                }
                gradE[k] = sum;
            }
            lm.getEmbeddingLayer().backprop(seq[i - 1], gradE, lr);

            // 5) Step progress bar
            pb.step();
            count++;
            if (logger.isDebugEnabled()) {
                logger.debug("  example {}/{} ▶ target token={}",
                             count, pb.getMax(), target);
            }
        }

        return new Result(loss, count);
    }

    /**
     * Count how many prediction examples are in the corpus
     * (sum of token-count-1 across all texts).
     */
    private static int countExamples(Tokenizer tok,
                                     List<Path> files) throws IOException {
        int total = 0;
        for (Path f : files) {
            String fn = f.getFileName().toString().toLowerCase();
            if (fn.endsWith(".txt")) {
                total += Math.max(0, tok.encode(Files.readString(f)).length - 1);
            } else if (fn.endsWith(".json")) {
                JsonNode root = MAPPER.readTree(f.toFile());
                JsonNode arr  = root.has("root") ? root.get("root") : root;
                for (JsonNode node : arr) {
                    if (node.has("text")) {
                        total += Math.max(0,
                            tok.encode(node.get("text").asText()).length - 1);
                    }
                }
            }
        }
        return total;
    }

    /** Simple softmax helper. */
    private static double[] softmax(double[] x) {
        double max = Arrays.stream(x).max().orElse(0);
        double sum = 0;
        double[] e = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            e[i] = Math.exp(x[i] - max);
            sum += e[i];
        }
        for (int i = 0; i < x.length; i++) e[i] /= sum;
        return e;
    }

    /** Internal holder for loss/count pair. */
    private static class Result {
        final double loss;
        final int count;
        Result(double l, int c) { loss = l; count = c; }
    }
}
