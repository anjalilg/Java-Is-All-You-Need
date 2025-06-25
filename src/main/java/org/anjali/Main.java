package org.anjali;

/**
 * Main entry point for a quick test of the TransformerEncoderLayer.
 * Creates a random dummy batch, feeds it through the encoder, and
 * prints the resulting shape.
 */
public class Main {
    public static void main(String[] args) {
        // Initialize one Transformer encoder layer with:
        // - input embedding dimension = 16
        // - number of attention heads = 4
        // - feed‑forward hidden dimension = 64
        TransformerEncoderLayer encoder = new TransformerEncoderLayer(16, 4, 64);

        // Create dummy input: 10 tokens each with 16‑dim embeddings
        double[][] input = new double[10][16];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                // Fill with random values between 0 and 1
                input[i][j] = Math.random();
            }
        }

        // Run the forward pass through the encoder layer
        double[][] output = encoder.forward(input);

        // Print out the shape of the output matrix
        System.out.println("Output shape: " + output.length + "x" + output[0].length);
    }
}