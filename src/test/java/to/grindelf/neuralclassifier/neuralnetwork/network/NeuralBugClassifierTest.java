package to.grindelf.neuralclassifier.neuralnetwork.network;

import org.junit.jupiter.api.Test;
import to.grindelf.neuralclassifier.domain.utils.Bug;
import to.grindelf.neuralclassifier.neuralnetwork.utils.FunctionName;
import to.grindelf.neuralclassifier.neuralnetwork.utils.FunctionsOrder;
import to.grindelf.neuralclassifier.neuralnetwork.utils.NeurShape;

import java.util.List;

import static to.grindelf.neuralclassifier.domain.data.utils.DatasetGeneratorKt.generateData;

class NeuralBugClassifierTest {

    private final List<Bug> dataset = generateData(100);
    private final List<Bug> testDataset = generateData(10);

    @Test
    void test() {
        var nbc = new NeuralBugClassifier(
                new NeurShape(2, List.of(5), 1),
                new FunctionsOrder(List.of(FunctionName.SIGMOID, FunctionName. SIGMOID))
        );

        nbc.train(dataset, 100, 0.001);

        var results = nbc.predict(testDataset);

        for (int i = 0; i < results.size(); i++)  {
            System.out.println(results.get(i).getValue() + " <- " + dataset.get(i).getType().getValue());
        }
    }

}