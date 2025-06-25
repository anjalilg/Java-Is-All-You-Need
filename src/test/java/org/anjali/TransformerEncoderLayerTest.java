package org.anjali;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * TransformerEncoderLayerTest
 *
 * Unit tests for TransformerEncoderLayer:
 * 1) testAddMethod verifies that the package-private add(...) method correctly sums two matrices.
 * 2) testForwardShape verifies that the forward(...) method preserves the input’s shape.
 */
public class TransformerEncoderLayerTest {

    /**
     * Test the add(...) helper method directly.
     * Creates two 2×2 matrices and checks elementwise addition.
     */
    @Test
    void testAddMethod() {
        double[][] a = {{1, 2}, {3, 4}};
        double[][] b = {{5, 6}, {7, 8}};
        TransformerEncoderLayer layer = new TransformerEncoderLayer(2, 1, 4);
        // call the package-private add(...) method
        double[][] sum = layer.add(a, b);
        assertArrayEquals(
            new double[]{6, 8},
            sum[0],
            1e-9,
            "First row should be [6, 8]"
        );
        assertArrayEquals(
            new double[]{10, 12},
            sum[1],
            1e-9,
            "Second row should be [10, 12]"
        );
    }

    /**
     * Test the forward(...) method’s shape preservation.
     * For an input of shape [4][8], the output must also be [4][8].
     */
    @Test
    void testForwardShape() {
        TransformerEncoderLayer layer = new TransformerEncoderLayer(8, 2, 16);
        double[][] input = new double[4][8];  // 4 tokens, embedding size 8
        double[][] out   = layer.forward(input);
        assertEquals(4, out.length,    "Output should have 4 rows");
        assertEquals(8, out[0].length, "Each output row should have length 8");
    }
}