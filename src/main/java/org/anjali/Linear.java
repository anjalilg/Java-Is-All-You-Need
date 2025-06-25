package org.anjali;

import java.util.Random;

/**
 * Linear
 *
 * Simple fully connected layer with weights and bias:
 * y = x·W + b
 */
public class Linear {
    // weight matrix [inDim][outDim]
    private final double[][] weights;
    // bias vector [outDim]
    private final double[] bias;

    /**
     * Initialize with random Gaussian weights (σ=0.02) and zero bias.
     *
     * @param in  input dimension
     * @param out output dimension
     */
    public Linear(int in, int out) {
        this.weights = new double[in][out];
        this.bias    = new double[out];
        Random rand = new Random();
        for (int i = 0; i < in; i++) {
            for (int j = 0; j < out; j++) {
                weights[i][j] = rand.nextGaussian() * 0.02;
            }
        }
        // bias[] is zero by default
    }

    /**
     * Forward pass: matrix multiply + bias.
     *
     * @param x input [seqLen][inDim]
     * @return output [seqLen][outDim]
     */
    public double[][] forward(double[][] x) {
        int n = x.length;
        int outDim = bias.length;
        double[][] y = new double[n][outDim];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < outDim; j++) {
                double sum = bias[j];
                for (int k = 0; k < x[0].length; k++) {
                    sum += x[i][k] * weights[k][j];
                }
                y[i][j] = sum;
            }
        }
        return y;
    }

    /** Accessor for weights (for saving/loading or backprop). */
    public double[][] getWeights() { return weights; }

    /** Accessor for bias vector. */
    public double[] getBias()      { return bias;    }

    /** Replace weights from external source (must match shape). */
    public void setWeights(double[][] newW) {
        for (int i = 0; i < weights.length; i++) {
            System.arraycopy(newW[i], 0, weights[i], 0, weights[0].length);
        }
    }

    /** Replace bias from external source (must match length). */
    public void setBias(double[] newB) {
        System.arraycopy(newB, 0, bias, 0, bias.length);
    }
}
