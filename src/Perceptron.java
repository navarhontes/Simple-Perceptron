package src;
// import java.util.Random;

public class Perceptron {
    private double[] weights;
    private double bias;
    private int inputSize; 
    
    public Perceptron(int inputSize) {
        this.inputSize = inputSize; 

        // Currently always initializas bias and weights randomly. 
        weights = new double[inputSize];
        //Random rand = new Random();
        for (int i = 0; i < inputSize; i++) {
            weights[i] = 0; 
            //weights[i] = rand.nextDouble();
        }
        //bias = rand.nextDouble();
        bias = 0;
    }

    public double getBias(){
        return this.bias; 
    }

    public double[] getWeights(){
        return this.weights; 
    }

    public int getInputSize(){
        return this.inputSize;
    }
    
    public int predict(double[] inputs) {
        double weightedSum = 0;
        // should either have correct size as precondition, or check and catch exception 
        for (int i = 0; i < inputs.length; i++) {
            weightedSum += inputs[i] * weights[i];
        }
        // don't know what's going on here 
        double output = weightedSum + bias > 0 ? 1 : 0;
        return (int) output;
    }
    
    public void train(double[][] inputs, int[] target, double learningRate, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            // should probably make it stop if no updates 
            for (int i = 0; i < inputs.length; i++) {
                int prediction = predict(inputs[i]);
                int error = target[i] - prediction;
                for (int j = 0; j < weights.length; j++) {
                    // is 0 1 nicer or + - nicer? 
                    weights[j] += learningRate * error * inputs[i][j];
                }
                bias += learningRate * error;
            }
        }
    }
}