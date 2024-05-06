package to.grindelf.neuralclassifier.neuralnetwork.network;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import to.grindelf.neuralclassifier.domain.utils.Bug;
import to.grindelf.neuralclassifier.neuralnetwork.utils.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static to.grindelf.neuralclassifier.neuralnetwork.utils.NeurMath.*;

public class NeuralBugClassifier implements Network {

    private final List<Neuron> inputLayer;
    private final List<Neuron> hiddenLayer;
    private final List<Neuron> outputLayer;
    private final FunctionsOrder functionsOrder;

    public NeuralBugClassifier(
            @NotNull NeurShape shape,
            @NotNull FunctionsOrder functionsOrder
    ) throws IllegalArgumentException {
        if (shape.getInputLayerSize() != 2) {
            throw new IllegalArgumentException("Input layer should be of size 2!");
        } else if (shape.getOutputLayerSize() != 1) {
            throw new IllegalArgumentException("Input layer should be of size 1!");
        }

        this.inputLayer = initLayer(shape.getInputLayerSize());
        if (shape.getHiddenLayersSizes().size() > 1) showToMuchHiddenLayersWarning();
        this.hiddenLayer = initLayer(shape.getHiddenLayersSizes().get(0));
        this.outputLayer = initLayer(shape.getOutputLayerSize());

        this.functionsOrder = functionsOrder;

        initWeightsAndThresholds();
    }

    /**
     * Trains neural network based on provided data for provided number of times (epochs),
     * with provided learning rate.
     *
     * @param data           data on which the training process is executed.
     * @param numberOfEpochs number of times network is trained.
     * @param learningRate   speed of network updates.
     *                       <p>
     * @return history of loss function changes
     * @throws IllegalArgumentException when data cannot be used for training.
     */
    @Override
    @NotNull
    public List<Double> train(
            @NotNull List<? extends NeurType> data,
            int numberOfEpochs,
            double learningRate
    ) throws IllegalArgumentException {
        List<Double> lossHistory = new ArrayList<>();
        List<Bug> bugs = new ArrayList<>();
        NeurClass target;
        double lossFunctionValue;
        double sumOfOutputLayerGradients;
        try {
            for (NeurType neurObject : data) {
                bugs.add((Bug) neurObject);
            }
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Provided data are incompatible for current classifier!");
        }

        for (int ignored = 0; ignored < numberOfEpochs; ignored++) {
            sumOfOutputLayerGradients = 0.0;
            Collections.shuffle(bugs);
            List<Double> lossFunctionsByCurrentEpoch = new ArrayList<>();
            for (Bug bug : bugs) {
                target = bug.getType();
                prepareValues(bug);

                forward();

                lossFunctionValue = crossEntropy(target.getValue(), outputLayer.get(0).getValue());
                lossFunctionsByCurrentEpoch.add(lossFunctionValue);

                sumOfOutputLayerGradients += lossFunctionValue * derivativeOfSigmoid(outputLayer.get(0).getSum());
            }

            backwards(sumOfOutputLayerGradients, learningRate);

            Double loss = meanLoss(lossFunctionsByCurrentEpoch);
            lossHistory.add(loss);
        }

        return lossHistory;
    }

    @NotNull
    private Double crossEntropy(
            double target,
            double actual
    ) {
        return -target * Math.log(actual) - (1 - target) * Math.log(1 - actual);
    }

    /**
     * Predicts classes of provided data samples.
     *
     * @param X data samples for which class will be predicted.
     * @return returns list of [NeurClass] values, which depict the class of the provided objects.
     * @throws IllegalArgumentException when X cannot be used for training.
     */
    @NotNull
    @Override
    public List<NeurClass> predict(
            @NotNull List<? extends NeurType> X
    ) throws IllegalArgumentException {
        List<NeurClass> classes = new ArrayList<>();

        List<Bug> bugs = new ArrayList<>();
        try {
            for (NeurType neurObject : X) {
                bugs.add((Bug) neurObject);
            }
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Provided data are incompatible for current classifier!");
        }

        for (Bug bug : bugs) {
            prepareValues(bug);
            forward();
            classes.add(new NeurClass(outputLayer.get(0).getValue()));
        }

        return classes;
    }

    private void showToMuchHiddenLayersWarning() {
        String message = "WARNING: There was more than one hidden layers size provided! " +
                "It is a reminder, that this classifier has only one hidden layer, so fist " +
                "of provided hidden layer sizes is picked.";

        System.out.println(message);
    }

    @NotNull
    private List<Neuron> initLayer(int layerSize) {
        List<Neuron> layer = new ArrayList<>();
        for (int i = 0; i < layerSize; i++) {
            layer.add(new Neuron());
        }

        return layer;
    }

    private void forward() {
        // input -> hidden
        for (int i = 0; i < hiddenLayer.size(); i++) {
            hiddenLayer.get(i).setSum(neuronSum(
                    inputLayer,
                    i,
                    hiddenLayer.get(i).getThreshold())
            );
            if (functionsOrder.getFirst() == FunctionName.SIGMOID) {
                hiddenLayer.get(i).setValue(sigmoid(hiddenLayer.get(i).getSum()));
            } else {
                hiddenLayer.get(i).setValue(relu(hiddenLayer.get(i).getSum()));
            }
        }

        // hidden -> output
        for (int i = 0; i < outputLayer.size(); i++) {
            outputLayer.get(i).setSum(neuronSum(
                    hiddenLayer,
                    i,
                    outputLayer.get(i).getThreshold())
            );
            if (functionsOrder.getSecond() == FunctionName.SIGMOID) {
                outputLayer.get(i).setValue(sigmoid(outputLayer.get(i).getSum()));
            } else {
                outputLayer.get(i).setValue(relu(outputLayer.get(i).getSum()));
            }
        }
    }

    @Contract(pure = true)
    private void backwards(@NotNull Double sumOfOutputLayerGradients, @NotNull Double learningRate) {
        // output -> hidden
        outputLayer.get(0).setLocalGradient(sumOfOutputLayerGradients);
        outputLayer.get(0).updateThreshold(learningRate * outputLayer.get(0).getLocalGradient() * outputLayer.get(0).getValue());

        for (Neuron hiddenNeuron : hiddenLayer) {
            hiddenNeuron.updateWeights(
                    0,
                    learningRate * outputLayer.get(0).getLocalGradient() * hiddenNeuron.getValue()
            );
            if (functionsOrder.getFirst() == FunctionName.SIGMOID) {
                hiddenNeuron.setLocalGradient(
                        outputLayer.get(0).getLocalGradient()
                                * hiddenNeuron.getWeights().get(0)
                                * derivativeOfSigmoid(hiddenNeuron.getSum())
                );

            } else {
                hiddenNeuron.setLocalGradient(
                        outputLayer.get(0).getLocalGradient()
                                * hiddenNeuron.getWeights().get(0)
                                * derivativeOfRelu(hiddenNeuron.getSum())
                );
            }
            hiddenNeuron.updateThreshold(learningRate * hiddenNeuron.getLocalGradient() * hiddenNeuron.getValue());
        }

        // hidden -> input
        for (Neuron inputNeuron : inputLayer) {
            for (int i = 0; i < hiddenLayer.size(); i++) {
                inputNeuron.updateWeights(
                        i,
                        learningRate * hiddenLayer.get(i).getLocalGradient() * inputNeuron.getValue()
                );
            }
        }
    }

    private void prepareValues(@NotNull Bug bug) {
        // input layer
        inputLayer.get(0).setValue(bug.getLength());
        inputLayer.get(1).setValue(bug.getWidth());
    }

    private void initWeightsAndThresholds() {

        Random randomizer = new Random(37);

        for (Neuron neuron : inputLayer) {
            neuron.setWeights(initRandomWeights(hiddenLayer.size(), randomizer));
        }

        List<Double> thresholds = new ArrayList<>();
        for (int ignored = 0; ignored < hiddenLayer.size() + outputLayer.size(); ignored++) {
            thresholds.add(randomizer.nextDouble(-1.0, 1.0));
        }

        // hidden layer
        for (int i = 0; i < hiddenLayer.size(); i++) {
            Neuron neuron = hiddenLayer.get(i);
            neuron.setWeights(initRandomWeights(outputLayer.size(), randomizer));
            neuron.setThreshold(thresholds.get(i));
        }

        // output layer
        for (int i = 0; i < outputLayer.size(); i++) {
            Neuron neuron = outputLayer.get(i);
            neuron.setThreshold(thresholds.get(i + hiddenLayer.size()));
        }

        // output layer has empty list of weights
        // and its other values will be initialized later or are initialized already
    }

    @NotNull
    @Contract(pure = true)
    private List<Double> initRandomWeights(int nextLayerSize, Random randomizer) {
        List<Double> weights = new ArrayList<>();
        for (int ignored = 0; ignored < nextLayerSize; ignored++) {
            weights.add(randomizer.nextDouble(0.0, 1.0));
        }

        return weights;
    }
}
