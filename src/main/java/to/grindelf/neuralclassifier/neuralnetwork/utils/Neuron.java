package to.grindelf.neuralclassifier.neuralnetwork.utils;

import org.jetbrains.annotations.NotNull;

import java.util.Collections;
import java.util.List;

public class Neuron {

    private double value;
    private List<Double> weights;
    private double error;
    private double sum;
    private double threshold;

    public Neuron() {
        this.value = 0.0;
        this.weights = Collections.emptyList();
        this.error = 0.0;
        this.sum = 0.0;
        this.threshold = 0.0;
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

    public void setWeights(List<Double> weights) {
        this.weights = weights;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
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
}
