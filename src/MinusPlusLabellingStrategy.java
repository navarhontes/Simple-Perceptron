package src;

public class MinusPlusLabellingStrategy implements LabellingStrategy{
    public boolean validLabel(int proposedLabel){
        return(proposedLabel == -1 || proposedLabel == 1); 
    }

    public String validLabels(){
        return("Valid labels: -1, 1"); 
    }
    
    public int convertToOutputLabel(double output){
        return (output > 0) ? 1 : -1;
    }

    public int[] convertToTrainingLabels(int[] labels){
        int[] newLabels = new int[labels.length]; 
        for(int i = 0; i < labels.length; i++){
            newLabels[i] = (labels[i] > 0) ? 1 : 0; 
        }
        return newLabels; 
    }
}
