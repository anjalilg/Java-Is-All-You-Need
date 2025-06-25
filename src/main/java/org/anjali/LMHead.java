package org.anjali;

import java.io.*;

/**
 * LMHead
 *
 * Projects hidden states -> vocabulary logits and
 * provides save/load/backprop for the last token.
 */
public class LMHead {
    // internal linear layer: embedDim -> vocabSize
    private final Linear proj;

    /**
     * @param embedDim  dimension of hidden vectors
     * @param vocabSize number of tokens
     */
    public LMHead(int embedDim, int vocabSize) {
        this.proj = new Linear(embedDim, vocabSize);
    }

    /**
     * Forward pass over a full sequence of hidden states.
     *
     * @param hidden [seqLen][embedDim]
     * @return logits [seqLen][vocabSize]
     */
    public double[][] forward(double[][] hidden) {
        return proj.forward(hidden);
    }

    /** Getter for the internal Linear (for weight access). */
    public Linear getProj() {
        return proj;
    }

    /** Save weights+bias to a file via Java serialization. */
    public void save(String path) throws IOException {
        try (ObjectOutputStream oos =
                     new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(proj.getWeights());
            oos.writeObject(proj.getBias());
        }
    }

    /**
     * Load from file and inject into an existing LMHead.
     *
     * @param head target head to modify
     * @param path file path containing serialized weights+bias
     */
    @SuppressWarnings("unchecked")
    public static void loadInto(LMHead head, String path)
            throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois =
                     new ObjectInputStream(new FileInputStream(path))) {
            double[][] w = (double[][]) ois.readObject();
            double[]   b = (double[])   ois.readObject();
            head.proj.setWeights(w);
            head.proj.setBias(b);
        }
    }

    /**
     * Backpropagate only the gradient for the final token:
     * W <- W - lr * (hLast * gradLogits),
     * b <- b - lr * gradLogits.
     *
     * @param hLast      hidden vector of last token
     * @param gradLogits gradient w.r.t. output logits
     * @param lr         learning rate
     */
    public void backpropLastToken(double[] hLast, double[] gradLogits, double lr) {
        double[][] W = proj.getWeights();
        double[]   B = proj.getBias();
        // update weights
        for (int k = 0; k < hLast.length; k++) {
            for (int j = 0; j < gradLogits.length; j++) {
                W[k][j] -= lr * gradLogits[j] * hLast[k];
            }
        }
        // update bias
        for (int j = 0; j < gradLogits.length; j++) {
            B[j] -= lr * gradLogits[j];
        }
    }
}
