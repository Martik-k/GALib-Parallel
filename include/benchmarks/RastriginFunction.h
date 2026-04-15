#ifndef RASTRIGIN_FUNCTION_H
#define RASTRIGIN_FUNCTION_H

#pragma once

/*
 * @file RastriginFunction.h
 * Global Minimum: The function has a unique global minimum at the origin
 * x = (0, 0, ..., 0), where f(x) = 0.
 * Local Minimum: It is highly multi-modal. Due to the cosine component, local
 * minima are located roughly at all points where each coordinate is an integer
 * (e.g., x_i ≈ ±1, ±2, ±3, ...).
 * Search Space: Usually evaluated on the hypercube x_i ∈ [-5.12, 5.12].
 */

#include "core/FitnessFunction.h"
#include <vector>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace galib::benchmark {

        template <typename GeneType = double>
        class RastriginFunction : public FitnessFunction<GeneType> {
        private:
            std::size_t dimensions_m;
			GeneType lower_bound_m;
			GeneType upper_bound_m;
            static constexpr double A = 10.0;

        public:
            explicit RastriginFunction(std::size_t dimensions, GeneType lower_bound, GeneType upper_bound)
                : dimensions_m(dimensions), lower_bound_m(lower_bound), upper_bound_m(upper_bound) {}

            double evaluate(const std::vector<GeneType>& genotype) const override {
                double sum = A * dimensions_m;

                for (GeneType gene : genotype) {
                    double x = static_cast<double>(gene);
                    sum += (x * x - A * std::cos(2.0 * std::numbers::pi * x));
                }

                return sum;
            }

            std::size_t size() const override {
                return dimensions_m;
            }

            GeneType getLowerBound(std::size_t) const override {
                return lower_bound_m;
            }

            GeneType getUpperBound(std::size_t) const override {
                return upper_bound_m;
            }
        };

}

#endif // RASTRIGIN_FUNCTION_H