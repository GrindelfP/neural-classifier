package to.grindelf.neuralclassifier.domain.utils;

import to.grindelf.neuralclassifier.neuralnetwork.utils.NeurClass;
import to.grindelf.neuralclassifier.neuralnetwork.utils.NeurType;

@NeurClassifiable
public class Bug implements NeurType {

    private final double length;
    private final double width;
    private final NeurClass type;

    public Bug(double length, double width, int type) {
        this.length = length;
        this.width = width;
        this.type = new NeurClass(type);
    }

    public Bug(double length, double width) {
        this.length = length;
        this.width = width;
        this.type = new NeurClass(-1);
    }

    public double getLength() {
        return length;
    }

    public double getWidth() {
        return width;
    }

    public NeurClass getType() {
        return type;
    }
}
