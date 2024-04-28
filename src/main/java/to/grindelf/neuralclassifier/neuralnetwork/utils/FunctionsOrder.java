package to.grindelf.neuralclassifier.neuralnetwork.utils;

import org.jetbrains.annotations.NotNull;

import java.util.List;

public class FunctionsOrder {

    private final FunctionName first;
    private final FunctionName second;

    public FunctionsOrder(@NotNull List<FunctionName> names) {
        this.first = names.get(0);
        this.second = names.get(1);
    }


    public FunctionName getFirst() {
        return first;
    }

    public FunctionName getSecond() {
        return second;
    }
}
