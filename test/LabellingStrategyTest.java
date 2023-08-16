package test;

import src.LabellingStrategy;
import src.MinusPlusLabellingStrategy;
import src.ZeroOneLabellingStrategy;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class LabellingStrategyTest {

    @Test
    public void testZeroOneOutput() {
        LabellingStrategy labellingStrategy = new ZeroOneLabellingStrategy(); 
        double[] outputs = {-1, 0, -10, 1, 10, 50};
        int[] expectedLabels = {0, 0, 0, 1, 1, 1};
        for (int i = 0; i < outputs.length; i++){
            assertEquals(expectedLabels[i], labellingStrategy.convertToOutputLabel(outputs[i]));
        }
    }

    @Test
    public void testPMOutput() {
        LabellingStrategy labellingStrategy = new MinusPlusLabellingStrategy(); 
        double[] outputs = {-1, 0, -10, 1, 10, 50};
        int[] expectedLabels = {-1, -1, -1, 1, 1, 1};
        for (int i = 0; i < outputs.length; i++){
            assertEquals(expectedLabels[i], labellingStrategy.convertToOutputLabel(outputs[i]));
        }
    }

    @Test
    public void testZeroOneTraining() {
        LabellingStrategy labellingStrategy = new ZeroOneLabellingStrategy(); 
        int[] labels = {-1, 0, 1};
        int[] expectedLabels = {0, 0, 1};

        assertArrayEquals(expectedLabels, labellingStrategy.convertToTrainingLabels(labels));
    }

    @Test
    public void testPMTraining() {
        LabellingStrategy labellingStrategy = new MinusPlusLabellingStrategy(); 
        int[] labels = {-1, 0, 1};
        int[] expectedLabels = {0, 0, 1};

        assertArrayEquals(expectedLabels, labellingStrategy.convertToTrainingLabels(labels));
    }
}

