package org.anjali;

import java.io.*;
import java.util.Random;

/**
 * Embedding
 *
 * Maps token IDs to fixed-dimension vectors, holds weights,
 * and supports forward lookup, saving/loading, and gradient updates.
 */
public class Embedding {
    private final double[][] weights; // [vocabSize][embedDim]

    /**
     * Initialize with Gaussian noise (σ=0.02).
     */
    public Embedding(int vocabSize, int embedDim) {
        weights = new double[vocabSize][embedDim];
        Random rnd = new Random();
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embedDim; j++) {
                weights[i][j] = rnd.nextGaussian() * 0.02;
            }
        }
    }

    /**
     * Forward: look up embedding vectors for a sequence of token IDs.
     *
     * @param tokens array of token IDs
     * @return [sequenceLength][embedDim] matrix
     */
    public double[][] forward(int[] tokens) {
        double[][] out = new double[tokens.length][weights[0].length];
        for (int i = 0; i < tokens.length; i++) {
            System.arraycopy(weights[tokens[i]], 0,
                             out[i], 0, weights[0].length);
        }
        return out;
    }

    /** Save the entire weight matrix to disk via Java serialization. */
    public void save(String path) throws IOException {
        try (ObjectOutputStream oos =
                     new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(weights);
        }
    }

    /** Load a saved weight matrix. */
    @SuppressWarnings("unchecked")
    public static double[][] load(String path)
            throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois =
                     new ObjectInputStream(new FileInputStream(path))) {
            return (double[][]) ois.readObject();
        }
    }

    /**
     * Overwrite current weights with loaded ones.
     * Throws if dimensions mismatch.
     */
    public void setWeights(double[][] newWeights) {
        if (newWeights.length != weights.length
         || newWeights[0].length != weights[0].length) {
            throw new IllegalArgumentException("Shape mismatch");
        }
        for (int i = 0; i < weights.length; i++) {
            System.arraycopy(newWeights[i], 0,
                             weights[i], 0, weights[0].length);
        }
    }

    /**
     * Stochastic gradient descent update for a single token.
     *
     * @param tokenId index of token whose embedding to update
     * @param gradE   gradient vector of same dimension
     * @param lr      learning rate
     */
    public void backprop(int tokenId, double[] gradE, double lr) {
        for (int k = 0; k < weights[0].length; k++) {
            weights[tokenId][k] -= lr * gradE[k];
        }
    }
}
