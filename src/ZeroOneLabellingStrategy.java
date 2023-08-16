package src;

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

    public int[] convertToTargetLabels(int[] labels){
        return labels;
    }
}
