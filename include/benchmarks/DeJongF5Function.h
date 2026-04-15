#ifndef DEJONG_F5_FUNCTION_H
#define DEJONG_F5_FUNCTION_H

#pragma once

#include "core/FitnessFunction.h"
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <string>
#include <cmath>

namespace galib::benchmark {

    template <typename GeneType = double>
    class DeJongF5Function : public FitnessFunction<GeneType> {
    private:
        static constexpr std::size_t DIMENSIONS_M = 2;
        static constexpr double LOWER_BOUND_M = -50;
        static constexpr double UPPER_BOUND_M = 50;

    public:
        DeJongF5Function() = default;
        ~DeJongF5Function() override = default;

        double evaluate(const std::vector<GeneType>& phenotype) const override {
            if (phenotype.size() != 2) {
                throw std::invalid_argument(
                    "De Jong F5 requires exactly 2 dimensions, but received " + 
                    std::to_string(phenotype.size())
                );
            }

            static constexpr double a[2][25] = {
                {-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32},
                {-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32}
            };

            const auto x = static_cast<double>(phenotype[0]);
            const auto y = static_cast<double>(phenotype[1]);

            double sum = 0.0;
            for (int i = 0; i < 25; ++i) {
                const double term1 = std::pow(x - a[0][i], 6);
                const double term2 = std::pow(y - a[1][i], 6);
                sum += 1.0 / (static_cast<double>(i + 1) + term1 + term2);
            }

            return 1.0 / (0.002 + sum);
        }

        [[nodiscard]] std::size_t size() const override {
            return DIMENSIONS_M;
        }

        GeneType getLowerBound(std::size_t dimension) const override {
            return static_cast<GeneType>(LOWER_BOUND_M);
        }

        GeneType getUpperBound(std::size_t dimension) const override {
            return static_cast<GeneType>(UPPER_BOUND_M);
        }
    };

} // namespace galib

#endif // DEJONG_F5_FUNCTION_H
