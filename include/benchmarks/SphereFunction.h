#ifndef SPHERE_FUNCTION_H
#define SPHERE_FUNCTION_H

#pragma once

/*
 * @file SphereFunction.h
 * - Global Minimum: It has a single global minimum at the origin
 * x = (0, 0, ..., 0), where f(x) = 0.
 * - Local Minimum: There are NO local minimum other than the global one.
 * - Shape: Geometrically, it represents an n-dimensional paraboloid.
 */

#include "core/FitnessFunction.h"
#include <vector>

namespace galib::benchmark {

    /**
     * @brief Implementation of the Sphere function.
     * 
     * A simple unimodal function representing an n-dimensional paraboloid.
     * Global minimum: f(0, ..., 0) = 0.
     * 
     * @tparam GeneType The numeric type of the genes.
     */
    template <typename GeneType = double>
        class SphereFunction : public FitnessFunction<GeneType> {
        private:
            std::size_t dimensions;
            GeneType lower_bound;
            GeneType upper_bound;
        public:
            /**
             * @brief Constructs the Sphere function with given dimensions and bounds.
             * @param dim  Problem dimensionality.
             * @param lb   Uniform lower bound.
             * @param ub   Uniform upper bound.
             */
            explicit SphereFunction(const std::size_t dim, GeneType lb, GeneType ub)
                : dimensions(dim), lower_bound(lb), upper_bound(ub) {}

            /**
             * @brief Evaluates the N-dimensional Sphere function.
             * @param phenotype Input vector.
             * @return Fitness value.
             */
            double evaluate(const std::vector<GeneType>& phenotype) const override {
                double sum = 0;
                for (GeneType gene : phenotype) {
                    sum += static_cast<double>(gene * gene);
                }
                return sum;
            }

            std::size_t size() const override { return dimensions;}

            GeneType getLowerBound(std::size_t) const override { return lower_bound; }
            GeneType getUpperBound(std::size_t) const override { return upper_bound; }
        };

}

#endif // SPHERE_FUNCTION_H
