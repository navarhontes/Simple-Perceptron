package src;

public class PerceptronFacade {
    private double[] weights;
    private double bias;
    private int inputSize; 

    private LabellingStrategy labellingStrategy; // Depend on the interface
    
    public PerceptronFacade(int inputSize) {
        this.inputSize = inputSize; 

        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = 0; 
        }
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

    public void printWeightsAndBias(){
        System.out.print("Weights: ");
        for (double weight : this.getWeights()) {
            System.out.print(weight + " ");
        }
        System.out.println("\nBias: " + this.getBias());
    }

    public String validLabelsAsString(){
        return this.labellingStrategy.validLabels(); 
    }

    public void setLabellingStrategy(LabellingStrategy newStrategy){
        this.labellingStrategy = newStrategy; 
    }

    public boolean isValidLabel(int label){
        return this.labellingStrategy.validLabel(label); 
    }
    
    public int predict(double[] inputs) {
        double weightedSum = 0;
        // should either have correct size as precondition, or check and catch exception 
        for (int i = 0; i < inputs.length; i++) {
            weightedSum += inputs[i] * weights[i];
        }
        // if (weightedSum + bias > 0), then set to 1; else 0
        int output = (weightedSum + bias > 0) ? 1 : 0;
        return output; 
    }

    public int labelInput(double[] inputs) {
        return this.labellingStrategy.convertToOutputLabel(this.predict(inputs)); 
    }
    
    /**
     * @param inputs array of length l (arbitrary) containing arrays of size inputSize that represent the inputs to be categorised 
     * @param target array of length l representing the desired outputs for those labels 
     * @param epochs number of times to run the perceptron algorithm 
     */
    public void train(double[][] inputs, int[] target, int epochs) {
        // weightsUpdated tracks whether any updates are actually happening or not, 
        // so that the training can stop if the categorizations are all correct 
        // and therefore not being updated. 
        boolean weightsUpdated = true; 
        target = this.labellingStrategy.convertToTargetLabels(target); 

        int epoch = 0; 
        while (epoch < epochs && weightsUpdated){
            weightsUpdated = false; 
            
            for (int i = 0; i < inputs.length; i++) {
                int prediction = predict(inputs[i]);
                int error = target[i] - prediction;
                        
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += error * inputs[i][j];
                }
                bias += error;

                if(error != 0){
                    weightsUpdated = true; 
                }
            }
            epoch++; 
        }
    }
}