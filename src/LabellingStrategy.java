public interface LabellingStrategy {
    boolean validLabel(int proposedLabel); 
    String validLabels(); 
    
    int convertToOutputLabel(double output);
    public int[] convertToTrainingLabels(int[] labels);
}
