#ifndef RASTRIGIN_FUNCTION_H
#define RASTRIGIN_FUNCTION_H

#pragma once

/*
 * @file RastriginFunction.h
 * Global Minimum: The function has a unique global minimum at the origin
 * x = (0, 0, ..., 0), where f(x) = 0.
 * Local Minimum: It is highly multi-modal. Due to the cosine component, local
 * minima are located roughly at all points where each coordinate is an integer
 * (e.g., x_i ≈ ±1, ±2, ±3, ...), creating a "egg-carton" surface.
 * Search Space: Usually evaluated on the hypercube x_i ∈ [-5.12, 5.12].
 */

#include "core/FitnessFunction.h"
#include <vector>
#include <cmath>
#include <cstddef>

namespace galib {
    namespace benchmark {
        template <typename GeneType = double>
        class RastriginFunction : public FitnessFunction<GeneType> {
        private:
            std::size_t dimensions_m;
            double A_m;

        public:
            explicit RastriginFunction(std::size_t dimensions = 3, double A = 10.0)
                : dimensions_m(dimensions), A_m(A) {}

            double evaluate(const std::vector<GeneType>& genotype) const override {
                const double PI = 3.14159265358979323846;
                double sum = A_m * dimensions_m;

                for (GeneType gene : genotype) {
                    double x = static_cast<double>(gene);
                    sum += (x * x - A_m * std::cos(2.0 * PI * x));
                }

                return sum;
            }

            std::size_t size() const override {
                return dimensions_m;
            }

            GeneType getLowerBound(std::size_t) const override {
                return static_cast<GeneType>(-5.12);
            }

            GeneType getUpperBound(std::size_t) const override {
                return static_cast<GeneType>(5.12);
            }
        };

    }
}

#endif // RASTRIGIN_FUNCTION_H