package to.grindelf.neuralclassifier.neuralnetwork.utils;

import org.jetbrains.annotations.NotNull;

import java.util.Collections;
import java.util.List;

public class Neuron {

    private double value;
    private List<Double> weights;
    private double sum;
    private double threshold;
    private double localGradient;


    public Neuron() {
        this.value = 0.0;
        this.weights = Collections.emptyList();
        this.sum = 0.0;
        this.threshold = 0.0;
        this.localGradient = 0.0;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    @NotNull
    public List<Double> getWeights() {
        return weights;
    }

    public void updateWeights(@NotNull Integer index, @NotNull Double value) {
        double currentWeight = this.weights.get(index);
        this.weights.set(index, currentWeight + value);
    }

    public void setWeights(List<Double> weights) {
        this.weights = weights;
    }

    public double getSum() {
        return sum;
    }

    public void setSum(double sum) {
        this.sum = sum;
    }

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }
    
    public void updateThreshold(double subtrahend) {
        this.threshold -= subtrahend;
    }

    public double getLocalGradient() {
        return localGradient;
    }

    public void setLocalGradient(double localGradient) {
        this.localGradient = localGradient;
    }
}
