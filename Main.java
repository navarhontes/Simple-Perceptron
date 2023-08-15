public class Main{
    public static void main(String[] args) {
        int[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] target = {0, 0, 0, 1};
        
        Perceptron perceptron = new Perceptron(2);
        perceptron.train(inputs, target, 0.1, 100);
        
        System.out.print("Weights: ");
        for (double weight : perceptron.getWeights()) {
            System.out.print(weight + " ");
        }
        System.out.println("\nBias: " + perceptron.getBias());
        
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptron.predict(inputs[i]);
            System.out.println("Input: [" + inputs[i][0] + ", " + inputs[i][1] + "], Target: " +
                    target[i] + ", Predicted: " + prediction);
        }
    }
}
