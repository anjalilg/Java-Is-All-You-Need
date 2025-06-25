package org.anjali;

import java.util.*;

/**
 * LanguageModel
 *
 * Combines a Tokenizer, Embedding layer, a stack of TransformerEncoderLayers,
 * and an LMHead to allow both training introspection and greedy text generation.
 */
public class LanguageModel {
    // Underlying tokenizer for text↔ID mapping
    private final Tokenizer tokenizer;
    // Embedding layer: maps token IDs to dense vectors
    private final Embedding embed;
    // Transformer encoder stack
    private final TransformerEncoderLayer[] layers;
    // Language modeling head: projects final hidden to vocab logits
    private final LMHead head;
    // Maximum generated sequence length
    private final int maxLen;

    /**
     * Construct a new LanguageModel.
     *
     * @param tokenizer existing Tokenizer with built vocab
     * @param embedDim  embedding dimension
     * @param numHeads  number of attention heads
     * @param hiddenDim feed-forward hidden size
     * @param numLayers number of encoder layers
     * @param maxLen    maximum generation length
     */
    public LanguageModel(Tokenizer tokenizer,
                         int embedDim, int numHeads, int hiddenDim,
                         int numLayers, int maxLen) {
        this.tokenizer = tokenizer;
        // initialize embedding matrix
        this.embed     = new Embedding(tokenizer.getVocabSize(), embedDim);
        // initialize LM head
        this.head      = new LMHead(embedDim, tokenizer.getVocabSize());
        this.maxLen    = maxLen;
        // create transformer layers
        this.layers    = new TransformerEncoderLayer[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layers[i] = new TransformerEncoderLayer(embedDim, numHeads, hiddenDim);
        }
    }

    /**
     * Greedy generation: encode prompt, then repeatedly pick the
     * highest-scoring next token until <EOS> or maxLen.
     *
     * @param prompt raw input text
     * @return generated completion (without BOS/EOS tokens)
     */
    public String generate(String prompt) {
        // 1) encode prompt: BOS + token IDs
        int[] init = tokenizer.encodePrompt(prompt);

        // 2) accumulate IDs in a dynamic list
        List<Integer> all = new ArrayList<>();
        for (int id : init) {
            all.add(id);
        }

        final int eosId = tokenizer.getEosId();

        // 3) autoregressive loop
        while (all.size() < maxLen) {
            // convert to int[]
            int[] seq = all.stream().mapToInt(i -> i).toArray();
            // 3a) embed the sequence
            double[][] hidden = embed.forward(seq);
            // 3b) pass through each transformer layer
            for (var layer : layers) {
                hidden = layer.forward(hidden);
            }
            // 3c) project to vocab logits
            double[] logits = head.forward(hidden)[hidden.length - 1];
            // 3d) pick argmax
            int next = argmax(logits);
            all.add(next);
            // stop on EOS
            if (next == eosId) break;
        }

        // 4) decode all IDs → space-joined tokens
        return tokenizer.decodeAll(all);
    }

    /** Helper to find index of max element in an array. */
    private int argmax(double[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[best]) {
                best = i;
            }
        }
        return best;
    }

    // Getters for training and introspection

    /** Retrieve the Tokenizer. */
    public Tokenizer getTokenizer()              { return tokenizer; }

    /** Retrieve the Embedding layer. */
    public Embedding getEmbeddingLayer()         { return embed; }

    /** Retrieve a copy of the Transformer layers. */
    public TransformerEncoderLayer[] getLayers() { return layers.clone(); }

    /** Retrieve the LM head. */
    public LMHead getOutputHead()                { return head; }
}
