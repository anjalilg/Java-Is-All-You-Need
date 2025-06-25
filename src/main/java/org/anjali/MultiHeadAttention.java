// MultiHeadAttention.java
package org.anjali;

import java.util.Arrays;

/**
 * MultiHeadAttention
 *
 * Implements scaled dot-product multi-head attention:
 * 1) Linear projections for queries, keys, values
 * 2) Split into multiple heads
 * 3) Scaled dot-product attention per head
 * 4) Concatenate heads and final linear projection
 */
public class MultiHeadAttention {
    private final int embedDim;   // total embedding dimension
    private final int numHeads;   // number of attention heads
    private final int headDim;    // dimension per head
    private final Linear wq, wk, wv, wo;  // projection layers

    /**
     * @param embedDim total embedding size (must be divisible by numHeads)
     * @param numHeads number of parallel attention heads
     */
    public MultiHeadAttention(int embedDim, int numHeads) {
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim  = embedDim / numHeads;
        // initialize the four linear maps
        this.wq = new Linear(embedDim, embedDim);
        this.wk = new Linear(embedDim, embedDim);
        this.wv = new Linear(embedDim, embedDim);
        this.wo = new Linear(embedDim, embedDim);
    }

    /**
     * Forward pass through multi-head attention.
     *
     * @param query [seqLen][embedDim]
     * @param key   [seqLen][embedDim]
     * @param value [seqLen][embedDim]
     * @return      [seqLen][embedDim] updated representations
     */
    public double[][] forward(double[][] query, double[][] key, double[][] value) {
        // 1) linear projections
        double[][] Q = wq.forward(query);
        double[][] K = wk.forward(key);
        double[][] V = wv.forward(value);

        // output accumulator
        double[][] output = new double[query.length][embedDim];

        // process each head
        for (int h = 0; h < numHeads; h++) {
            int start = h * headDim, end = start + headDim;
            // slice out this head's portion
            double[][] Qh = slice(Q, start, end);
            double[][] Kh = slice(K, start, end);
            double[][] Vh = slice(V, start, end);

            // compute attention scores: Qh × Khᵀ
            double[][] scores = matmul(Qh, transpose(Kh));
            // scale and normalize
            scores = softmax(scale(scores, 1.0 / Math.sqrt(headDim)));
            // weighted sum over values
            double[][] headOut = matmul(scores, Vh);

            // write back into output matrix
            for (int i = 0; i < output.length; i++) {
                System.arraycopy(headOut[i], 0, output[i], start, headDim);
            }
        }

        // final linear projection
        return wo.forward(output);
    }

    /** Slice columns [start, end) from each row of x. */
    private double[][] slice(double[][] x, int start, int end) {
        double[][] out = new double[x.length][end - start];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(x[i], start, out[i], 0, end - start);
        }
        return out;
    }

    /** Transpose a 2D matrix. */
    private double[][] transpose(double[][] x) {
        int rows = x.length, cols = x[0].length;
        double[][] t = new double[cols][rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t[j][i] = x[i][j];
        return t;
    }

    /** Standard matrix multiplication. */
    private double[][] matmul(double[][] a, double[][] b) {
        int n = a.length, m = b[0].length, k = a[0].length;
        double[][] out = new double[n][m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                for (int l = 0; l < k; l++)
                    out[i][j] += a[i][l] * b[l][j];
        return out;
    }

    /** Scale each element by factor. */
    private double[][] scale(double[][] x, double factor) {
        for (int i = 0; i < x.length; i++)
            for (int j = 0; j < x[0].length; j++)
                x[i][j] *= factor;
        return x;
    }

    /** Row-wise softmax for attention weights. */
    private double[][] softmax(double[][] x) {
        double[][] out = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            double max = Arrays.stream(x[i]).max().getAsDouble();
            double sum = 0.0;
            for (int j = 0; j < x[i].length; j++) {
                out[i][j] = Math.exp(x[i][j] - max);
                sum += out[i][j];
            }
            for (int j = 0; j < x[i].length; j++) {
                out[i][j] /= sum;
            }
        }
        return out;
    }

    // Getters for introspection
    public int getEmbedDim() { return embedDim; }
    public int getNumHeads() { return numHeads; }
    public int getHeadDim()  { return headDim;  }
}
