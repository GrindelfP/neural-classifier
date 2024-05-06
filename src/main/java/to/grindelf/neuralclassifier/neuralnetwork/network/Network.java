package to.grindelf.neuralclassifier.neuralnetwork.network;

import org.jetbrains.annotations.NotNull;
import to.grindelf.neuralclassifier.domain.utils.Loss;
import to.grindelf.neuralclassifier.neuralnetwork.utils.NeurClass;
import to.grindelf.neuralclassifier.neuralnetwork.utils.NeurType;

import java.util.List;

/**
 * Interface which contains basic network functionality.
 */
public interface Network {

    /**
     * Trains neural network based on provided data for provided number of times (epochs),
     * with provided learning rate.
     *
     * @param data data on which the training process is executed.
     * @param numberOfEpochs number of times network is trained.
     * @param learningRate speed of network updates.
     * <p>
     * @return history of loss function changes
     *
     * @throws IllegalArgumentException when data cannot be used for training.
     */
    public List<Double> train(@NotNull List<? extends NeurType> data, int numberOfEpochs, double learningRate) throws IllegalArgumentException;


    /**
     * Predicts classes of provided data samples.
     * @param X data samples for which class will be predicted.
     * @return returns list of [NeurClass] values, which depict the class of the provided objects.
     * @throws IllegalArgumentException when X cannot be used for training.
     */
    @NotNull
    public List<NeurClass> predict(@NotNull List<? extends NeurType> X) throws IllegalArgumentException;

}
