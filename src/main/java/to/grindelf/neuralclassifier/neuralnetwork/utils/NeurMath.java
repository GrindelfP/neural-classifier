package to.grindelf.neuralclassifier.neuralnetwork.utils;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import to.grindelf.neuralclassifier.domain.utils.Loss;

import java.util.List;

import static java.lang.Math.exp;
import static java.lang.Math.max;

public class NeurMath {

    @NotNull
    public static Double meanLoss(@NotNull List<Double> list) {
        double loss = 0.0;

        if (!list.isEmpty()) {
            double sum = 0.0;
            for (double num : list) {
                sum += num;
            }
            loss = sum / list.size();
        }

        return loss;
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    public static double derivativeOfSigmoid(double x) {
        return sigmoid(x) * (1.0 - sigmoid(x));
    }

    public static double relu(double x) {
        return max(0, x);
    }

    public static double derivativeOfRelu(double x) {
        double derivative = 0.0;
        if (x > 0.0) derivative = 1.0;

        return derivative;
    }

    @Contract(pure = true)
    public static double neuronSum(@NotNull List<Neuron> neurons, int targetNeuronIndex, double targetThreshold) {
        double sum = 0.0;
        for (Neuron neuron : neurons) {
            sum += neuron.getValue() * neuron.getWeights().get(targetNeuronIndex);
        }

        return sum + targetThreshold;
    }
}
