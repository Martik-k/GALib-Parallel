#ifndef HEAVY_TRIG_FUNCTION_H
#define HEAVY_TRIG_FUNCTION_H

#pragma once

#include "core/FitnessFunction.h"

#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace galib::benchmark {

/**
 * @brief A computationally intensive trigonometric benchmark function.
 * 
 * Designed to test the library's parallel scaling capabilities by introducing 
 * a heavy workload in the fitness evaluation via repeated sine/cosine iterations.
 * 
 * @tparam GeneType The numeric type of the genes.
 */
template <typename GeneType = double>
class HeavyTrigFunction : public FitnessFunction<GeneType> {
private:
    std::size_t dimensions_m;
    GeneType lower_bound_m;
    GeneType upper_bound_m;
    std::size_t repeat_count_m;

public:
    /**
     * @brief Constructs the benchmark.
     * @param dimensions   Number of dimensions.
     * @param lower_bound  Uniform lower bound.
     * @param upper_bound  Uniform upper bound.
     * @param repeat_count Number of internal iterations per dimension (default: 64).
     */
    explicit HeavyTrigFunction(std::size_t dimensions,
                               GeneType lower_bound,
                               GeneType upper_bound,
                               std::size_t repeat_count = 64)
        : dimensions_m(dimensions),
          lower_bound_m(lower_bound),
          upper_bound_m(upper_bound),
          repeat_count_m(repeat_count) {}

    /**
     * @brief Evaluates the heavy trigonometric sum.
     * @param phenotype Input vector.
     * @return Fitness value.
     */
    double evaluate(const std::vector<GeneType>& phenotype) const override {
        if (phenotype.size() != dimensions_m) {
            throw std::invalid_argument(
                "HeavyTrig requires exactly " + std::to_string(dimensions_m) +
                " dimensions, but received " + std::to_string(phenotype.size())
            );
        }

        double sum = 0.0;
        for (std::size_t i = 0; i < phenotype.size(); ++i) {
            const double x = static_cast<double>(phenotype[i]);
            double state = x;
            for (std::size_t r = 1; r <= repeat_count_m; ++r) {
                const double factor = static_cast<double>(r);
                const double angle = factor * 0.017 + x * 0.031;
                state = std::sin(state + angle)
                      + std::cos(state - angle * 0.5)
                      + std::exp(-0.001 * state * state)
                      + 0.0005 * state * state;
            }
            sum += state * state + 0.1 * x * x;
        }
        return sum;
    }

    [[nodiscard]] std::size_t size() const override { return dimensions_m; }
    [[nodiscard]] std::string name() const override { return "HeavyTrig"; }

    GeneType getLowerBound(std::size_t) const override {
        return lower_bound_m;
    }

    GeneType getUpperBound(std::size_t) const override {
        return upper_bound_m;
    }
};

} // namespace galib::benchmark

#endif // HEAVY_TRIG_FUNCTION_H