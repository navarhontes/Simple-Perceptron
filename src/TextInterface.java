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
            System.out.println("2. Make Prediction");
            System.out.println("3. Exit");
            System.out.print("Enter your choice: ");

            try {
                int choice = scanner.nextInt();
                scanner.nextLine(); // Consume newline
                
                switch (choice) {
                    case 0:
                        perceptronFacade = createPerceptronFacade(scanner);
                        break;
                    case 1:
                        trainPerceptron(scanner, perceptronFacade);
                        break;
                    case 2:
                        makePrediction(scanner, perceptronFacade);
                        break;
                    case 3:
                        System.out.println("Exiting the program. Goodbye!");
                        scanner.close();
                        return;
                    default:
                        System.out.println("Invalid choice. Please choose again.");
                }
            } catch (InputMismatchException e) {
                System.out.println("Invalid input. Please enter a valid integer.");
                scanner.nextLine(); // Consume invalid input
            }
        }
    }

    private static PerceptronFacade createPerceptronFacade(Scanner scanner) {
        // Create and return an instance of PerceptronFacade with necessary parameters
        while (true){
            try {
                int inputSize = scanner.nextInt();
                scanner.nextLine(); // Consume newline
                return new PerceptronFacade(inputSize); 
            } catch (InputMismatchException e) {
                System.out.println("Invalid input. Please enter a valid integer.");
                scanner.nextLine(); // Consume invalid input
            }
        }
    }

    private static void trainPerceptron(Scanner scanner, PerceptronFacade perceptronFacade) {
        System.out.print("Enter the number of training data points: ");
        int numDataPoints = scanner.nextInt();
        scanner.nextLine(); // Consume newline
        
        int inputDimension = perceptronFacade.getInputSize();
        int[][] inputs = new int[numDataPoints][inputDimension];
        int[] targets = new int[numDataPoints];
        
        System.out.println("Enter training data in the format [input1 input2 ... inputN target]:");
        for (int i = 0; i < numDataPoints; i++) {
            System.out.print("Data " + (i + 1) + ": ");
            String[] data = scanner.nextLine().split(" ");
            
            if (data.length != inputDimension + 1) {
                System.out.println("Invalid input. Please enter " + (inputDimension + 1) + " values.");
                i--; // Retry this data point
                continue;
            }
            
            for (int j = 0; j < inputDimension; j++) {
                inputs[i][j] = Integer.parseInt(data[j]);
            }
            targets[i] = Integer.parseInt(data[inputDimension]);
        }
        
        System.out.print("Enter learning rate: ");
        double learningRate = scanner.nextDouble();
        scanner.nextLine(); // Consume newline
        
        System.out.print("Enter number of epochs: ");
        int epochs = scanner.nextInt();
        scanner.nextLine(); // Consume newline
        
        perceptronFacade.train(inputs, targets, learningRate, epochs);
        
        System.out.println("Training completed.");
    }

    private static void makePrediction(Scanner scanner, PerceptronFacade perceptronFacade) {
        int inputDimension = perceptronFacade.getInputSize();
        int[] inputs = new int[inputDimension];
        
        System.out.println("Enter input values for prediction:");
        for (int i = 0; i < inputDimension; i++) {
            System.out.print("Input " + (i + 1) + ": ");
            inputs[i] = scanner.nextInt();
        }
        
        int prediction = perceptronFacade.predict(inputs);
        System.out.println("Prediction: " + prediction);
    }
    
}


