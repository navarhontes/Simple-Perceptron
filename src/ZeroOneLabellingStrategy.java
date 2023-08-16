public class ZeroOneLabellingStrategy implements LabellingStrategy{
    public boolean validLabel(int proposedLabel){
        return(proposedLabel == 0 || proposedLabel == 1); 
    }
    
    public String validLabels(){
        return("Valid labels: 0, 1"); 
    }
    
    public int convertToOutputLabel(double output){
        return (output > 0) ? 1 : 0;
    }

    public int[] convertToTrainingLabels(int[] labels){
        int[] newLabels = new int[labels.length]; 
        for(int i = 0; i < labels.length; i++){
            newLabels[i] = (labels[i] > 0) ? 1 : 0; 
        }
        return newLabels; 
    }
}
