package to.grindelf.neuralclassifier.domain.data.utils;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;
import static org.junit.jupiter.api.Assertions.*;

class DataReaderTest {

    private final String BUGS_DATASET_PATH = "src/test/resources/bugs-dataset.csv";

    @Test
    void getBugsDataSetFrom() {
        var bugs = DataReader.getBugsDataSetFrom(BUGS_DATASET_PATH);

        assertThat(bugs.size()).isEqualTo(200);
    }
}