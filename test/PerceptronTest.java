package test;

import src.PerceptronFacade;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class PerceptronTest {

    @Test
    public void testANDGate() {
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] target = {0, 0, 0, 1};

        PerceptronFacade perceptron = new PerceptronFacade(2);
        perceptron.train(inputs, target, 100);

        double[] weights = perceptron.getWeights();
        double bias = perceptron.getBias();

        assertArrayEquals(new double[]{2, 1}, weights, 0.01);
        assertEquals(-2, bias, 0.01);

        int[] expectedPredictions = {0, 0, 0, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptron.predict(inputs[i]);
            assertEquals(expectedPredictions[i], prediction);
        }
    }

    @Test
    public void testANDGate3DimentionalInput() {
        double[][] inputs = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
        int[] target = {0, 0, 0, 0, 0, 0, 0, 1};

        PerceptronFacade perceptron = new PerceptronFacade(3);
        perceptron.train(inputs, target, 100);
        
        int[] expectedPredictions = {0, 0, 0, 0, 0, 0, 0, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptron.predict(inputs[i]);
            assertEquals(expectedPredictions[i], prediction);
        }
    }

    @Test
    public void testORGate() {
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] target = {0, 0, 0, 1};

        PerceptronFacade perceptron = new PerceptronFacade(2);
        perceptron.train(inputs, target, 100);

        int[] expectedPredictions = {0, 0, 0, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptron.predict(inputs[i]);
            assertEquals(expectedPredictions[i], prediction);
        }
    }

    @Test
    public void testMIT6036Week2Homework3() {
        // Borrowed from MIT OpenCourseWare 6.036
        double[][] inputs = {{-3, 2}, {-1, 1}, {-1, -1}, {2, 2}, {1, -1}};
        int[] target = {1, 0, 0, 0, 0};

        PerceptronFacade perceptron = new PerceptronFacade(2);
        perceptron.train(inputs, target, 100);

        int[] expectedPredictions = {1, 0, 0, 0, 0};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptron.predict(inputs[i]);
            assertEquals(expectedPredictions[i], prediction);
        }
    }

    @Test
    public void testXORGate() {
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] target = {0, 1, 1, 0};

        PerceptronFacade perceptron = new PerceptronFacade(2);
        perceptron.train(inputs, target, 100);

        int[] expectedPredictions = {0, 0, 0, 1};
        int[] actualPredictions = new int[4]; 
        for (int i = 0; i < inputs.length; i++) {
            actualPredictions[i] = perceptron.predict(inputs[i]);
        }
        // Since this case is not linearly separable
        assertNotEquals(expectedPredictions, actualPredictions); 
    }

    @Test
    public void testMIT6036Week2Homework51() {
        // Borrowed from MIT OpenCourseWare 6.036
        double[][] inputs = {{1, -1}, {1, 1}, {2, -1}, {2, 1}};
        int[] target = {0, 1, 1, 0};

        PerceptronFacade perceptron = new PerceptronFacade(2);
        perceptron.train(inputs, target, 100);

        int[] expectedPredictions = {0, 1, 1, 0};
        int[] actualPredictions = new int[4]; 
        for (int i = 0; i < inputs.length; i++) {
            actualPredictions[i] = perceptron.predict(inputs[i]);
        }
        // Since this case is not linearly separable
        assertNotEquals(expectedPredictions, actualPredictions); 
    }
}

