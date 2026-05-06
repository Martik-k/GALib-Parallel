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
         * @param dims   Number of dimensions.
         * @param lower  Lower bound applied to all dimensions.
         * @param upper  Upper bound applied to all dimensions.
         * @param func   Evaluation callable (lambda, function pointer, std::function).
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
         * @param dims    Number of dimensions.
         * @param lowers  Vector of lower bounds per dimension.
         * @param uppers  Vector of upper bounds per dimension.
         * @param func    Evaluation callable.
         * @throws std::invalid_argument if bounds vector sizes do not match dims.
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

        /**
         * @brief Evaluates the fitness of a candidate solution using the wrapped function.
         * @param phenotype Gene values of the individual.
         * @return Fitness score.
         */
        [[nodiscard]] double evaluate(const std::vector<GeneType>& phenotype) const override {
            return evaluator_m(phenotype);
        }

        /**
         * @brief Returns the problem dimensionality.
         * @return Number of dimensions.
         */
        [[nodiscard]] std::size_t size() const override {
            return dimensions_m;
        }

        /**
         * @brief Gets the lower bound for a specific dimension.
         * @param dimension Index of the dimension.
         * @return Lower bound value.
         */
        [[nodiscard]] GeneType getLowerBound(std::size_t dimension) const override {
            return lower_bounds_m.at(dimension);
        }

        /**
         * @brief Gets the upper bound for a specific dimension.
         * @param dimension Index of the dimension.
         * @return Upper bound value.
         */
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
