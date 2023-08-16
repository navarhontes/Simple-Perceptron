package test;

import src.PerceptronFacade;
import src.MinusPlusLabellingStrategy;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;

public class PerceptronFacadeTest {

    @Test
    public void testANDGate() {
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] target = {0, 0, 0, 1};

        PerceptronFacade perceptronFacade = new PerceptronFacade(2);
        perceptronFacade.train(inputs, target, 100);

        double[] weights = perceptronFacade.getWeights();
        double bias = perceptronFacade.getBias();

        assertArrayEquals(new double[]{2, 1}, weights, 0.01);
        assertEquals(-2, bias, 0.01);

        int[] expectedPredictions = {0, 0, 0, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPredictions[i], prediction);
        }

        perceptronFacade.setLabellingStrategy(new MinusPlusLabellingStrategy());
        perceptronFacade.createNewPerceptron(2);
        perceptronFacade.train(inputs, target, 100);

        weights = perceptronFacade.getWeights();
        bias = perceptronFacade.getBias();

        assertArrayEquals(new double[]{2, 1}, weights, 0.01);
        assertEquals(-2, bias, 0.01);

        int[] expectedPMPredictions = {-1,-1, -1, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPMPredictions[i], prediction);
        }
    }

    @Test
    public void testANDGate3DimentionalInput() {
        double[][] inputs = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
        int[] target = {0, 0, 0, 0, 0, 0, 0, 1};

        PerceptronFacade perceptronFacade = new PerceptronFacade(3);
        perceptronFacade.train(inputs, target, 100);
        
        int[] expectedPredictions = {0, 0, 0, 0, 0, 0, 0, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPredictions[i], prediction);
        }

        perceptronFacade.setLabellingStrategy(new MinusPlusLabellingStrategy());
        perceptronFacade.createNewPerceptron(3);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPMPredictions = {-1, -1, -1, -1, -1, -1, -1, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPMPredictions[i], prediction);
        }
    }

    @Test
    public void testORGate() {
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] target = {0, 0, 0, 1};

        PerceptronFacade perceptronFacade = new PerceptronFacade(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPredictions = {0, 0, 0, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPredictions[i], prediction);
        }

        perceptronFacade.setLabellingStrategy(new MinusPlusLabellingStrategy());
        perceptronFacade.createNewPerceptron(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPMPredictions = {-1, -1, -1, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPMPredictions[i], prediction);
        }
    }

    @Test
    public void testMIT6036Week2Homework3() {
        // Borrowed from MIT OpenCourseWare 6.036
        double[][] inputs = {{-3, 2}, {-1, 1}, {-1, -1}, {2, 2}, {1, -1}};
        int[] target = {1, 0, 0, 0, 0};

        PerceptronFacade perceptronFacade = new PerceptronFacade(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPredictions = {1, 0, 0, 0, 0};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPredictions[i], prediction);
        }

        perceptronFacade.setLabellingStrategy(new MinusPlusLabellingStrategy());
        perceptronFacade.createNewPerceptron(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPMPredictions = {1, -1, -1, -1, -1};

        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPMPredictions[i], prediction);
        }
    }

    @Test
    public void testXORGate() {
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] target = {0, 1, 1, 0};

        PerceptronFacade perceptronFacade = new PerceptronFacade(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPredictions = {0, 0, 0, 1};
        int[] actualPredictions = new int[4]; 
        for (int i = 0; i < inputs.length; i++) {
            actualPredictions[i] = perceptronFacade.predict(inputs[i]);
        }
        // Since this case is not linearly separable
        assertFalse(Arrays.equals(expectedPredictions, actualPredictions)); 

        perceptronFacade.setLabellingStrategy(new MinusPlusLabellingStrategy());
        perceptronFacade.createNewPerceptron(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPMPredictions = {-1, -1, -1, 1};
        
        for (int i = 0; i < inputs.length; i++) {
            actualPredictions[i] = perceptronFacade.predict(inputs[i]);
        }
        // Since this case is not linearly separable
        assertFalse(Arrays.equals(expectedPMPredictions, actualPredictions)); 
    }

    @Test
    public void testMIT6036Week2Homework51() {
        // Borrowed from MIT OpenCourseWare 6.036
        double[][] inputs = {{1, -1}, {1, 1}, {2, -1}, {2, 1}};
        int[] target = {0, 1, 1, 0};

        PerceptronFacade perceptronFacade = new PerceptronFacade(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPredictions = {0, 1, 1, 0};
        int[] actualPredictions = new int[4]; 
        for (int i = 0; i < inputs.length; i++) {
            actualPredictions[i] = perceptronFacade.predict(inputs[i]);
        }
        // Since this case is not linearly separable
        assertFalse(Arrays.equals(expectedPredictions, actualPredictions)); 
        
        int[] expectedPMPredictions = {-1, 1, 1, -1};
        perceptronFacade.setLabellingStrategy(new MinusPlusLabellingStrategy());
        perceptronFacade.createNewPerceptron(2);
        perceptronFacade.train(inputs, target, 100);

        for (int i = 0; i < inputs.length; i++) {
            actualPredictions[i] = perceptronFacade.predict(inputs[i]);
        }
        // Since this case is not linearly separable
        assertFalse(Arrays.equals(expectedPMPredictions, actualPredictions)); 
    }

    @Test
    public void testPMLabelsANDGate() {
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] target = {-1, -1, -1, 1};

        PerceptronFacade perceptronFacade = new PerceptronFacade(2);
        perceptronFacade.train(inputs, target, 100);

        double[] weights = perceptronFacade.getWeights();
        double bias = perceptronFacade.getBias();

        assertArrayEquals(new double[]{2, 1}, weights, 0.01);
        assertEquals(-2, bias, 0.01);

        int[] expectedPredictions = {0, 0, 0, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPredictions[i], prediction);
        }

        perceptronFacade.setLabellingStrategy(new MinusPlusLabellingStrategy());
        perceptronFacade.createNewPerceptron(2);
        perceptronFacade.train(inputs, target, 100);

        weights = perceptronFacade.getWeights();
        bias = perceptronFacade.getBias();

        assertArrayEquals(new double[]{2, 1}, weights, 0.01);
        assertEquals(-2, bias, 0.01);

        int[] expectedPMPredictions = {-1, -1, -1, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPMPredictions[i], prediction);
        }
    }

    @Test
    public void testPMLabelsANDGate3DimentionalInput() {
        double[][] inputs = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
        int[] target = {-1, -1, -1, -1, -1, -1, -1, 1};

        PerceptronFacade perceptronFacade = new PerceptronFacade(3);
        perceptronFacade.train(inputs, target, 100);
        
        int[] expectedPredictions = {0, 0, 0, 0, 0, 0, 0, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPredictions[i], prediction);
        }

        perceptronFacade.setLabellingStrategy(new MinusPlusLabellingStrategy());
        perceptronFacade.createNewPerceptron(3);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPMPredictions = {-1, -1, -1, -1, -1, -1, -1, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPMPredictions[i], prediction);
        }
    }

    @Test
    public void testPMLabelsORGate() {
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] target = {-1, -1, -1, 1};

        PerceptronFacade perceptronFacade = new PerceptronFacade(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPredictions = {0, 0, 0, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPredictions[i], prediction);
        }

        perceptronFacade.setLabellingStrategy(new MinusPlusLabellingStrategy());
        perceptronFacade.createNewPerceptron(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPMPredictions = {-1, -1, -1, 1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPMPredictions[i], prediction);
        }
    }

    @Test
    public void testPMLabelsMIT6036Week2Homework3() {
        // Borrowed from MIT OpenCourseWare 6.036
        double[][] inputs = {{-3, 2}, {-1, 1}, {-1, -1}, {2, 2}, {1, -1}};
        int[] target = {1, -1, -1, -1, -1};

        PerceptronFacade perceptronFacade = new PerceptronFacade(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPredictions = {1, 0, 0, 0, 0};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPredictions[i], prediction);
        }

        perceptronFacade.setLabellingStrategy(new MinusPlusLabellingStrategy());
        perceptronFacade.createNewPerceptron(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPMPredictions = {1, -1, -1, -1, -1};
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            assertEquals(expectedPMPredictions[i], prediction);
        }
    }

    @Test
    public void testPMLabelsXORGate() {
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] target = {-1, 1, 1, -1};

        PerceptronFacade perceptronFacade = new PerceptronFacade(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPredictions = {0, 0, 0, 1};
        int[] actualPredictions = new int[4]; 
        for (int i = 0; i < inputs.length; i++) {
            actualPredictions[i] = perceptronFacade.predict(inputs[i]);
        }
        // Since this case is not linearly separable
        assertFalse(Arrays.equals(expectedPredictions, actualPredictions)); 

        perceptronFacade.setLabellingStrategy(new MinusPlusLabellingStrategy());
        perceptronFacade.createNewPerceptron(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPMPredictions = {-1, -1, -1, 1};
        for (int i = 0; i < inputs.length; i++) {
            actualPredictions[i] = perceptronFacade.predict(inputs[i]);
        }
        // Since this case is not linearly separable
        assertFalse(Arrays.equals(expectedPMPredictions, actualPredictions)); 
    }

    @Test
    public void testPMLabelsMIT6036Week2Homework51() {
        // Borrowed from MIT OpenCourseWare 6.036
        double[][] inputs = {{1, -1}, {1, 1}, {2, -1}, {2, 1}};
        int[] target = {-1, 1, 1, -1};

        PerceptronFacade perceptronFacade = new PerceptronFacade(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPredictions = {0, 1, 1, 0};
        int[] actualPredictions = new int[4]; 
        for (int i = 0; i < inputs.length; i++) {
            actualPredictions[i] = perceptronFacade.predict(inputs[i]);
        }
        // Since this case is not linearly separable
        assertFalse(Arrays.equals(expectedPredictions, actualPredictions)); 

        perceptronFacade.setLabellingStrategy(new MinusPlusLabellingStrategy());
        perceptronFacade.createNewPerceptron(2);
        perceptronFacade.train(inputs, target, 100);

        int[] expectedPMPredictions = {0, 1, 1, 0};
        for (int i = 0; i < inputs.length; i++) {
            actualPredictions[i] = perceptronFacade.predict(inputs[i]);
        }
        // Since this case is not linearly separable
        assertFalse(Arrays.equals(expectedPMPredictions, actualPredictions)); 
    }

    }

