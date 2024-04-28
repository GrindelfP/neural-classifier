package to.grindelf.neuralclassifier.domain.utils;

public class Loss {

    private double functionValue;

    public Loss(double functionValue) {
        this.functionValue = functionValue;
    }

    public double getFunctionValue() {
        return functionValue;
    }

    public void setFunctionValue(double value) {
        this.functionValue = value;
    }
}
