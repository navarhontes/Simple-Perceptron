package src;

import java.util.Scanner;
import java.util.InputMismatchException;

public class TextInterface {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("Perceptron Program Text Interface");
        System.out.println("--------------------------------");

        // To make sure we don't do anything else without a perceptron to work with 
        PerceptronFacade perceptronFacade = createPerceptronFacade(scanner);

        while (true) {
            System.out.println("0. Initialise new Perceptron");
            System.out.println("1. Train Perceptron");
            System.out.println("2. Make prediction");
            System.out.println("3. Choose labelling convention");
            System.out.println("4. Exit");
            System.out.print("Enter your choice: ");

            int choice = getValidIntInput(scanner);
            
            switch (choice) {
                case 0:
                    createPerceptron(scanner, perceptronFacade);
                    break;
                case 1:
                    trainPerceptron(scanner, perceptronFacade);
                    break;
                case 2:
                    makePrediction(scanner, perceptronFacade);
                    break;
                case 3:
                    chooseLabellingConvention(scanner, perceptronFacade);
                    break;
                case 4:
                    System.out.println("Exiting the program. Goodbye!");
                    scanner.close();
                    return;
                default:
                    System.out.println("Invalid choice. Please choose again.");
            }
        }
    }

    private static PerceptronFacade createPerceptronFacade(Scanner scanner) {
        // Create and return an instance of PerceptronFacade with necessary parameters
        System.out.print("Enter the input dimension (must be a positive integer!): ");
        int inputSize = getValidIntInput(scanner);
        while(inputSize <= 0){
            System.out.println("Invalid input. Please enter a valid integer.");
            inputSize = getValidIntInput(scanner);
        }
        return new PerceptronFacade(inputSize); 
    }

    private static PerceptronFacade createPerceptron(Scanner scanner, PerceptronFacade perceptronFacade) {
        // Create and return an instance of PerceptronFacade with necessary parameters
        System.out.print("Enter the input dimension (must be a positive integer!): ");
        int inputSize = getValidIntInput(scanner);
        while(inputSize <= 0){
            System.out.println("Invalid input. Please enter a valid integer.");
            inputSize = getValidIntInput(scanner);
        }
        perceptronFacade.createNewPerceptron(inputSize); 
        return perceptronFacade; 
    }

    private static void trainPerceptron(Scanner scanner, PerceptronFacade perceptronFacade) {
        System.out.print("Enter the number of training data points (must be a positive integer!): ");
        int numDataPoints = getValidIntInput(scanner);
        while(numDataPoints <= 0){
            System.out.println("Invalid input. Please enter a valid integer.");
            numDataPoints = getValidIntInput(scanner);
        }
        
        int inputDimension = perceptronFacade.getInputSize();
        double[][] inputs = new double[numDataPoints][inputDimension];
        int[] targets = new int[numDataPoints];
        
        System.out.println("Enter training data in the following format:");
        System.out.println("input1 input2 ... inputN"); 
        System.out.println("target"); 
        System.out.println("Example:"); 
        System.out.println("1 1"); 
        System.out.println("1"); 

        for (int i = 0; i < numDataPoints; i++) {
            boolean validInput = false; 
            
            while(!validInput){
                System.out.print("Data " + (i + 1) + ": ");
                String[] data = scanner.nextLine().split(" ");
                
                if (data.length != inputDimension) {
                    System.out.println("Invalid input. Please enter " + (inputDimension) + " values.");
                } else{
                    try{
                        for (int j = 0; j < inputDimension; j++) {
                            inputs[i][j] = Double.parseDouble(data[j]);
                        }
                        validInput = true; 
                    } catch(InputMismatchException e){
                        System.out.println("Invalid input. Please enter a valid decimal.");
                    }
                }
            }
            targets[i] = getValidIntInput(scanner); 
            while(!perceptronFacade.isValidLabel(targets[i])){
                System.out.println(perceptronFacade.validLabelsAsString()); 
                targets[i] = getValidIntInput(scanner); 
            }
        }
        
        System.out.print("Enter number of epochs (must be a positive integer!): ");
        int epochs = getValidIntInput(scanner);
        while(epochs <= 0){
            System.out.println("Invalid input. Please enter a valid integer.");
            epochs = getValidIntInput(scanner);
        }
        
        perceptronFacade.train(inputs, targets, epochs);
        
        System.out.println("Training completed.");
        perceptronFacade.printWeightsAndBias();
        for (int i = 0; i < inputs.length; i++) {
            int prediction = perceptronFacade.predict(inputs[i]);
            System.out.println("Input: [" + inputs[i][0] + ", " + inputs[i][1] + "], Target: " +
                    targets[i] + ", Predicted: " + prediction);
        }
    }

    private static void makePrediction(Scanner scanner, PerceptronFacade perceptronFacade) {
        int inputDimension = perceptronFacade.getInputSize();
        double[] inputs = new double[inputDimension];
        
        System.out.println("Enter input values for prediction:");
        for (int i = 0; i < inputDimension; i++) {
            System.out.print("Input " + (i + 1) + ": ");
            inputs[i] = getValidDoubleInput(scanner); 
        }
        
        int prediction = perceptronFacade.predict(inputs);
        System.out.println("Prediction: " + prediction);
    }

    private static void chooseLabellingConvention(Scanner scanner, PerceptronFacade perceptronFacade){
        while (true) {
            System.out.println("0. Valid labels: 0, 1");
            System.out.println("1. Valid labels: -1, 1");
            System.out.print("Enter your choice: ");

            int choice = getValidIntInput(scanner);
            
            switch (choice) {
                case 0:
                    perceptronFacade.setLabellingStrategy(new ZeroOneLabellingStrategy());
                    return; 
                case 1:
                    perceptronFacade.setLabellingStrategy(new MinusPlusLabellingStrategy());
                    return; 
                default:
                    System.out.println("Invalid choice. Please choose again.");
            }
        }
    }

    public static int getValidIntInput(Scanner scanner){
        int input; 
        while (true){
            try {
                input = scanner.nextInt();
                scanner.nextLine(); // Consume newline
                return input; 
            } catch (InputMismatchException e) {
                System.out.println("Invalid input. Please enter a valid integer.");
                scanner.nextLine(); // Consume invalid input
            }
        }
    }

    public static double getValidDoubleInput(Scanner scanner){
        double input; 
        while (true){
            try {
                input = scanner.nextDouble();
                scanner.nextLine(); // Consume newline
                return input; 
            } catch (InputMismatchException e) {
                System.out.println("Invalid input. Please enter a valid integer.");
                scanner.nextLine(); // Consume invalid input
            }
        }
    }
    
}


