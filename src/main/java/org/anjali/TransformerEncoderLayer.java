package org.anjali;

/**
 * TransformerEncoderLayer
 *
 * Single encoder block: Multi-Head Attention + Add&Norm,
 * followed by Feed-Forward + Add&Norm.
 */
public class TransformerEncoderLayer {
    private final MultiHeadAttention mha;
    private final FeedForward ff;
    private final LayerNorm norm1;
    private final LayerNorm norm2;

    /**
     * @param embedDim hidden size
     * @param numHeads number of attention heads
     * @param hiddenDim feed-forward inner size
     */
    public TransformerEncoderLayer(int embedDim, int numHeads, int hiddenDim) {
        this.mha   = new MultiHeadAttention(embedDim, numHeads);
        this.ff    = new FeedForward(embedDim, hiddenDim);
        this.norm1 = new LayerNorm(embedDim);
        this.norm2 = new LayerNorm(embedDim);
    }

    /**
     * Forward through:
     * 1) Self-attention -> add residual -> norm
     * 2) Feed-forward -> add residual -> norm
     *
     * @param x input [seqLen][embedDim]
     * @return  output [seqLen][embedDim]
     */
    public double[][] forward(double[][] x) {
        // 1) attention block
        double[][] attn = mha.forward(x, x, x);
        double[][] x1   = norm1.forward(add(x, attn));

        // 2) feed-forward block
        double[][] ffOut = ff.forward(x1);
        double[][] x2    = norm2.forward(add(x1, ffOut));

        return x2;
    }

    /**
     * Elementwise addition of two matrices.
     * Package-private so tests in same package can call.
     */
    double[][] add(double[][] a, double[][] b) {
        int n = a.length, d = a[0].length;
        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                out[i][j] = a[i][j] + b[i][j];
            }
        }
        return out;
    }

    // expose internals if necessary
    public MultiHeadAttention getMha()       { return mha; }
    public FeedForward       getFeedForward(){ return ff;  }
}




