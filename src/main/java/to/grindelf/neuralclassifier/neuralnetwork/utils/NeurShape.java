package to.grindelf.neuralclassifier.neuralnetwork.utils;

import org.jetbrains.annotations.NotNull;

import java.util.List;

public class NeurShape {

    private final int inputLayerSize;
    private final List<Integer> hiddenLayersSizes;
    private final int outputLayerSize;

    public NeurShape(int inputLayerSize, List<Integer> hiddenLayersSizes, int outputLayerSize) {
        this.inputLayerSize = inputLayerSize;
        this.hiddenLayersSizes = hiddenLayersSizes;
        this.outputLayerSize = outputLayerSize;
    }

    public int getInputLayerSize() {
        return inputLayerSize;
    }

    @NotNull
    public List<Integer> getHiddenLayersSizes() {
        return hiddenLayersSizes;
    }

    public int getOutputLayerSize() {
        return outputLayerSize;
    }
}
