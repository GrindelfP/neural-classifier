package to.grindelf.neuralclassifier.domain.data.utils

import to.grindelf.neuralclassifier.domain.utils.Bug

fun generateData(size: Int): List<Bug> {
    val data = mutableListOf<Bug>()
    for (i in 1..size) {
        if (Math.random() < 0.5) {
            var length = 2 + Math.random() * 3
            length = String.format("%.2f", length).toDouble()
            val width = length
            data.add(Bug(length, width, 0))
        } else {
            var length = 7 + Math.random() * 8
            length = String.format("%.2f", length).toDouble()
            var width = 0.5 + Math.random() * 1
            width = String.format("%.2f", width).toDouble()
            data.add(Bug(length, width, 1))
        }
    }
    return data
}