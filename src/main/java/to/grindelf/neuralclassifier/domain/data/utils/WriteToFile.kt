package to.grindelf.neuralclassifier.domain.data.utils

import java.io.File

object WriteToFile {
    fun writeDoubleVector(data: List<Double>) {
        val file = File("losses.txt")
        file.bufferedWriter().use { out ->
            data.forEach { value ->
                out.write("$value\n")
            }
        }
    }
}