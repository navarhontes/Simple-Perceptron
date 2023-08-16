package src;

public class PerceptronFacade {
    private Perceptron perceptron; 
    private LabellingStrategy labellingStrategy; // Depend on the interface
    
    public PerceptronFacade(int inputSize) {
        this.perceptron = new Perceptron(inputSize); 
        this.labellingStrategy = new ZeroOneLabellingStrategy(); 
    }

    public double getBias(){
        return this.perceptron.getBias(); 
    }

    public double[] getWeights(){
        return this.perceptron.getWeights(); 
    }

    public int getInputSize(){
        return this.perceptron.getInputSize();
    }

    public void printWeightsAndBias(){
        System.out.print("Weights: ");
        for (double weight : this.getWeights()) {
            System.out.print(weight + " ");
        }
        System.out.println("\nBias: " + this.getBias());
    }

    public void setLabellingStrategy(LabellingStrategy newStrategy){
        this.labellingStrategy = newStrategy; 
    }

    /**
     * @return a string in the form of "Valid labels: x, y"
     */
    public String validLabelsAsString(){
        return this.labellingStrategy.validLabels(); 
    }

    /**
     * @return true if a label fits into the labelling strategy; false otherwise. 
     */
    public boolean isValidLabel(int label){
        return this.labellingStrategy.validLabel(label); 
    }
    
    public int predict(double[] inputs) {
        return this.perceptron.predict(inputs); 
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
        target = this.labellingStrategy.convertToTrainingLabels(target); 
        this.perceptron.train(inputs, target, epochs); 
    }
}