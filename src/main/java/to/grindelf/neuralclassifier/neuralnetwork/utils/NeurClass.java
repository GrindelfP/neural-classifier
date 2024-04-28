package to.grindelf.neuralclassifier.neuralnetwork.utils;

/**
 * Contains data about class of the object - the result of neural classification.
 */
public class NeurClass {

    private final double value;

    public NeurClass(double value) {
        this.value = value > 0.5 ? 1 : 0;
    }

    public double getValue() {
        return value;
    }
}
