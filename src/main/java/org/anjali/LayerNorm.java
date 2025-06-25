package org.anjali;

/**
 * LayerNorm
 *
 * Applies per-token layer normalization over the last dimension:
 * (x - mean)/sqrt(var + ε).
 */
public class LayerNorm {
    // size of hidden dimension
    private final int dim;

    /**
     * @param dim number of features per token
     */
    public LayerNorm(int dim) {
        this.dim = dim;
    }

    /**
     * Normalize each token vector independently.
     *
     * @param x input matrix [seqLen][dim]
     * @return normalized output [seqLen][dim]
     */
    public double[][] forward(double[][] x) {
        double[][] out = new double[x.length][dim];
        for (int i = 0; i < x.length; i++) {
            // compute mean
            double sum = 0;
            for (int j = 0; j < dim; j++) {
                sum += x[i][j];
            }
            double mean = sum / dim;
            // compute variance
            double var = 0;
            for (int j = 0; j < dim; j++) {
                double diff = x[i][j] - mean;
                var += diff * diff;
            }
            var /= dim;
            double std = Math.sqrt(var + 1e-5);

            // apply normalization
            for (int j = 0; j < dim; j++) {
                out[i][j] = (x[i][j] - mean) / std;
            }
        }
        return out;
    }

    /** Return the normalized dimension. */
    public int getDim() { return dim; }
}
