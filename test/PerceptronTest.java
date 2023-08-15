package test;

import src.Perceptron;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class PerceptronTest {

    @Test
    public void testANDGate() {
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] target = {0, 0, 0, 1};

        Perceptron perceptron = new Perceptron(2);
        perceptron.train(inputs, target, 0.1, 100);

        double[] weights = perceptron.getWeights();
        double bias = perceptron.getBias();

        assertArrayEquals(new double[]{0.2, 0.2}, weights, 0.01);
        assertEquals(-0.3, bias, 0.01);

        int[] expectedPredictions = {0, 0, 0, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptron.predict(inputs[i]);
            assertEquals(expectedPredictions[i], prediction);
        }
    }
}

