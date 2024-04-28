package to.grindelf.neuralclassifier.neuralnetwork.network;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import to.grindelf.neuralclassifier.domain.utils.Bug;
import to.grindelf.neuralclassifier.domain.utils.Loss;
import to.grindelf.neuralclassifier.neuralnetwork.utils.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static to.grindelf.neuralclassifier.neuralnetwork.utils.NeurMath.*;

public class NeuralBugClassifier implements Network {

    private final List<Neuron> inputLayer;
    private final List<Neuron> hiddenLayer;
    private final List<Neuron> outputLayer;
    private final FunctionsOrder functionsOrder;

    public NeuralBugClassifier(@NotNull NeurShape shape, @NotNull FunctionsOrder functionsOrder) throws IllegalArgumentException {
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
    public List<Loss> train(@NotNull List<? extends NeurType> data, int numberOfEpochs, double learningRate) throws IllegalArgumentException {
        List<Loss> lossHistory = new ArrayList<>();
        List<Bug> bugs = new ArrayList<>();
        NeurClass target;
        try {
            for (NeurType neurObject : data) {
                bugs.add((Bug) neurObject);
            }
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Provided data are incompatible for current classifier!");
        }

        for (int ignored = 0; ignored < numberOfEpochs; ignored++) {
            List<Double> lossFunctionsForCurrentEpoch = new ArrayList<>();
            for (Bug bug : bugs) {
                target = bug.getType();
                prepareValues(bug);
                forward();
                lossFunctionsForCurrentEpoch.add(backwards(target));
                updates(learningRate);
            }
            Loss loss = meanLoss(lossFunctionsForCurrentEpoch);
            lossHistory.add(loss);
            System.out.println("Current loss is " + loss.getFunctionValue());
        }

        return lossHistory;
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
    public List<NeurClass> predict(@NotNull List<? extends NeurType> X) throws IllegalArgumentException {
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
        for (int i = 0; i < hiddenLayer.size(); i++) {
            if (functionsOrder.getFirst() == FunctionName.SIGMOID) {
                hiddenLayer.get(i).setValue(sigmoid(neuronValue(inputLayer, i, hiddenLayer.get(i).getThreshold())));
            } else {
                hiddenLayer.get(i).setValue(relu(neuronValue(inputLayer, i, hiddenLayer.get(i).getThreshold())));
            }
        }
        for (int i = 0; i < outputLayer.size(); i++) {
            if (functionsOrder.getFirst() == FunctionName.SIGMOID) {
                outputLayer.get(i).setValue(sigmoid(neuronValue(hiddenLayer, i, outputLayer.get(i).getThreshold())));
            } else {
                outputLayer.get(i).setValue(relu(neuronValue(hiddenLayer, i, outputLayer.get(i).getThreshold())));
            }
        }
    }

    @Contract(pure = true)
    private double backwards(@NotNull NeurClass target) {
        outputLayer.get(0).setError(Math.pow(target.getValue() - outputLayer.get(0).getValue(), 2));

        for (Neuron neuron : hiddenLayer) {
            if (functionsOrder.getFirst() == FunctionName.SIGMOID) {
                neuron.setError(derivativeOfSigmoid(neuron.getValue()) * neuron.getWeights().get(0) * outputLayer.get(0).getError());
            } else {
                neuron.setError(derivativeOfRelu(neuron.getValue()) * neuron.getWeights().get(0) * outputLayer.get(0).getError());
            }
        }

        return outputLayer.get(0).getError();
    }

    private void updates(double learningRate) {

        for (int i = 0; i < hiddenLayer.size(); i++) {
            for (Neuron neuron : inputLayer) {
                double iWeight = neuron.getWeights().get(i);
                neuron.getWeights().set(i, iWeight + neuron.getValue() * hiddenLayer.get(i).getError() * learningRate);
            }
        }

        for (int i = 0; i < outputLayer.size(); i++) {
            for (Neuron neuron : hiddenLayer) {
                double iWeight = neuron.getWeights().get(i);
                neuron.getWeights().set(i, iWeight + neuron.getValue() * outputLayer.get(i).getError() * learningRate);
            }
        }

    }

    private void prepareValues(@NotNull Bug bug) {
        // input layer
        inputLayer.get(0).setValue(bug.getLength());
        inputLayer.get(1).setValue(bug.getWidth());
        for (Neuron neuron : inputLayer) {
            neuron.setWeights(initRandomWeights(hiddenLayer.size()));
            neuron.setThreshold(Math.random());
        }

        // hidden layer
        for (Neuron neuron : hiddenLayer) {
            neuron.setWeights(initRandomWeights(hiddenLayer.size()));
            neuron.setThreshold(Math.random());
        }

        // output layer has empty list of weights and its other values
        // will be initialized later or are initialized already

    }

    @NotNull
    @Contract(pure = true)
    private List<Double> initRandomWeights(int nextLayerSize) {
        List<Double> weights = new ArrayList<>();
        Random randomizer = (new Random());
        for (int ignored = 0; ignored < nextLayerSize; ignored++) {
            weights.add(randomizer.nextGaussian());
        }

        return weights;
    }
}
