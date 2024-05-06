package to.grindelf.neuralclassifier.domain.data.utils;

import kotlin.Triple;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import to.grindelf.neuralclassifier.domain.utils.Bug;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataReader {

    @NotNull
    @Contract(pure = true)
    public static List<Bug> getBugsDataSetFrom(@NotNull String path) {
        List<Bug> bugs = new ArrayList<>();

        List<Triple<Double, Double, Integer>> bugsDataRaw = readCsv(path);
        for (Triple<Double, Double, Integer> triple : bugsDataRaw) {
            double x1 = triple.getFirst();
            double x2 = triple.getSecond();
            int y = triple.getThird();
            bugs.add(new Bug(x1, x2, y));
        }

        return bugs;
    }

    @NotNull
    private static List<Triple<Double, Double, Integer>> readCsv(String filePath) {
        List<Triple<Double, Double, Integer>> list = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                Double first = Double.parseDouble(values[0]);
                Double second = Double.parseDouble(values[1]);
                Integer third = Integer.parseInt(values[2]);
                list.add(new Triple<>(first, second, third));
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return list;
    }

}
