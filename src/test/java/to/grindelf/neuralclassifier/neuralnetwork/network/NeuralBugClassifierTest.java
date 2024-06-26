package to.grindelf.neuralclassifier.neuralnetwork.network;

import org.junit.jupiter.api.Test;
import to.grindelf.neuralclassifier.domain.data.utils.WriteToFile;
import to.grindelf.neuralclassifier.domain.utils.Bug;
import to.grindelf.neuralclassifier.neuralnetwork.utils.FunctionName;
import to.grindelf.neuralclassifier.neuralnetwork.utils.FunctionsOrder;
import to.grindelf.neuralclassifier.neuralnetwork.utils.NeurShape;

import java.util.Collections;
import java.util.List;

import static to.grindelf.neuralclassifier.domain.data.utils.DataReader.getBugsDataSetFrom;

class NeuralBugClassifierTest {

    private final String BUGS_DATASET_PATH = "src/test/resources/bugs-dataset.csv";
    private final String BUGS_TEST_DATASET_PATH = "src/test/resources/bugs-test-dataset.csv";

    private final List<Bug> dataset = getBugsDataSetFrom(BUGS_DATASET_PATH);
    private final List<Bug> testDataset = getBugsDataSetFrom(BUGS_TEST_DATASET_PATH);

    @Test
    void test() {
        var nbc = new NeuralBugClassifier(
                new NeurShape(2, List.of(20), 1),
                new FunctionsOrder(List.of(FunctionName.RELU, FunctionName. SIGMOID))
        );

        var history = nbc.train(dataset, 800, 0.00028);
        var bestEpochLoss = Collections.min(history);

        WriteToFile.INSTANCE.writeDoubleVector(history);

        var results = nbc.predict(testDataset);
        var accuracy = 0.0;

        System.out.println("The best loss is " + bestEpochLoss + " at " + history.indexOf(bestEpochLoss));

        System.out.println("Predicted <- Actual");

        for (int i = 0; i < results.size(); i++)  {
            System.out.println(results.get(i).getValue() + " <- " + testDataset.get(i).getType().getValue());
            if (results.get(i).getValue() == testDataset.get(i).getType().getValue()) accuracy += 1.0;
        }
        accuracy /= results.size();

        System.out.println("Accuracy: " + accuracy);
    }

}