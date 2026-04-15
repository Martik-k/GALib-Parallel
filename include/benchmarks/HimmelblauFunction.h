#ifndef HIMMELBLAU_FUNCTION_H
#define HIMMELBLAU_FUNCTION_H

#pragma once

#include "core/FitnessFunction.h"
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <string>

namespace galib::benchmark {

    template <typename GeneType = double>
    class HimmelblauFitness : public FitnessFunction<GeneType> {
    private:
        static constexpr std::size_t DIMENSIONS_M = 2;
        static constexpr double LOWER_BOUND_M = -5.0;
        static constexpr double UPPER_BOUND_M = 5.0;

    public:
        HimmelblauFitness() = default;

        double evaluate(const std::vector<GeneType>& phenotype) const override {
            if (phenotype.size() != DIMENSIONS_M) {
                throw std::invalid_argument(
                    "Himmelblau's function requires exactly 2 dimensions, but received " + 
                    std::to_string(phenotype.size())
                );
            }

            const auto x = static_cast<double>(phenotype[0]);
            const auto y = static_cast<double>(phenotype[1]);

            const double term1 = (x * x) + y - 11.0;
            const double term2 = x + (y * y) - 7.0;

            return (term1 * term1) + (term2 * term2);
        }

        [[nodiscard]] std::size_t size() const{
            return DIMENSIONS_M;
        }

        GeneType getLowerBound(const std::size_t dimension) const override {
            return static_cast<GeneType>(LOWER_BOUND_M);
        }

        GeneType getUpperBound(const std::size_t dimension) const override {
            return static_cast<GeneType>(UPPER_BOUND_M);
        }
    };

} // namespace galib

#endif // HIMMELBLAU_FUNCTION_H