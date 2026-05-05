#ifndef FUNCTIONAL_FITNESS_H
#define FUNCTIONAL_FITNESS_H

#pragma once

#include "core/FitnessFunction.h"
#include <functional>
#include <vector>
#include <cstddef>
#include <utility>

namespace galib {

    /**
     * @brief A wrapper for FitnessFunction that allows using a lambda or std::function.
     * 
     * This class eliminates the need for users to create a new class for every custom
     * optimization problem. It supports both uniform bounds (same for all dimensions)
     * and non-uniform bounds.
     * 
     * @tparam GeneType The type of the genes (default is double).
     */
    template <typename GeneType = double>
    class FunctionalFitness : public FitnessFunction<GeneType> {
    public:
        using EvalFunc = std::function<double(const std::vector<GeneType>&)>;

        /**
         * @brief Constructor for uniform bounds.
         * @param dims Number of dimensions.
         * @param lower Lower bound for all dimensions.
         * @param upper Upper bound for all dimensions.
         * @param func The evaluation function (lambda, function pointer, etc.).
         */
        FunctionalFitness(
            std::size_t dims,
            GeneType lower,
            GeneType upper,
            EvalFunc func
        ) : dimensions_m(dims),
            lower_bounds_m(dims, lower),
            upper_bounds_m(dims, upper),
            evaluator_m(std::move(func)) {}

        /**
         * @brief Constructor for non-uniform bounds.
         * @param dims Number of dimensions.
         * @param lowers Vector of lower bounds per dimension.
         * @param uppers Vector of upper bounds per dimension.
         * @param func The evaluation function.
         */
        FunctionalFitness(
            std::size_t dims,
            std::vector<GeneType> lowers,
            std::vector<GeneType> uppers,
            EvalFunc func
        ) : dimensions_m(dims),
            lower_bounds_m(std::move(lowers)),
            upper_bounds_m(std::move(uppers)),
            evaluator_m(std::move(func)) {
            if (lower_bounds_m.size() != dims || upper_bounds_m.size() != dims) {
                throw std::invalid_argument("Bounds vectors must match dimensions size");
            }
        }

        [[nodiscard]] double evaluate(const std::vector<GeneType>& phenotype) const override {
            return evaluator_m(phenotype);
        }

        [[nodiscard]] std::size_t size() const override {
            return dimensions_m;
        }

        [[nodiscard]] GeneType getLowerBound(std::size_t dimension) const override {
            return lower_bounds_m.at(dimension);
        }

        [[nodiscard]] GeneType getUpperBound(std::size_t dimension) const override {
            return upper_bounds_m.at(dimension);
        }

    private:
        std::size_t dimensions_m;
        std::vector<GeneType> lower_bounds_m;
        std::vector<GeneType> upper_bounds_m;
        EvalFunc evaluator_m;
    };

} // namespace galib

#endif // FUNCTIONAL_FITNESS_H
