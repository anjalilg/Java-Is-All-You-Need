package org.anjali;

/**
 * FeedForward
 *
 * Two-layer fully connected network with ReLU activation in between.
 * Project up to hiddenDim then back down to embedDim.
 */
public class FeedForward {
    private final Linear fc1;
    private final Linear fc2;

    /**
     * @param embedDim size of input/output embeddings
     * @param hiddenDim size of intermediate layer
     */
    public FeedForward(int embedDim, int hiddenDim) {
        this.fc1 = new Linear(embedDim, hiddenDim);
        this.fc2 = new Linear(hiddenDim, embedDim);
    }

    /**
     * Forward pass:
     * 1) linear -> ReLU -> 2) linear
     */
    public double[][] forward(double[][] x) {
        double[][] out = fc1.forward(x);  // project up
        relu(out);                        // in-place ReLU
        return fc2.forward(out);         // project down
    }

    /** In-place ReLU activation. */
    private void relu(double[][] m) {
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[0].length; j++) {
                m[i][j] = Math.max(0.0, m[i][j]);
            }
        }
    }

    // Getters for introspection/testing
    public Linear getFc1() { return fc1; }
    public Linear getFc2() { return fc2; }
}