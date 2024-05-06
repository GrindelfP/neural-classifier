package to.grindelf.neuralclassifier.domain.utils

import kotlin.random.Random

object KotlinRandomizer {

    fun getRandomVectorBetween(from: Double, to: Double, seed: Int = 37, times: Int = 1): List<Double> {
        val randomVector = mutableListOf<Double>()
        val random = Random(seed)
        repeat(times) {
            randomVector.add(random.nextDouble(from, to))
        }

        return randomVector
    }

    fun getRandomBetween(from: Double, to: Double, seed: Int = 37): Double = Random(seed).nextDouble(from, to)
}