#ifndef FITNESS_FUNCTION_H
#define FITNESS_FUNCTION_H

#pragma once

#include <string>
#include <vector>
#include <cstddef>

namespace galib {

    /**
     * @brief Abstract base class for objective functions to be minimized.
     * 
     * Users should inherit from this class to define their own optimization problems,
     * or use FunctionalFitness to wrap a lambda.
     * 
     * @tparam GeneType The numeric type of the genes (default: double).
     */
    template <typename GeneType = double>
    class FitnessFunction {
    public:
        virtual ~FitnessFunction() = default;

        /**
         * @brief Evaluates the fitness of a candidate solution.
         * 
         * Lower return values indicate better fitness (minimization convention).
         * 
         * @param phenotype Gene values of the individual to evaluate.
         * @return Fitness score.
         */
        virtual double evaluate(const std::vector<GeneType>& phenotype) const = 0;

        /**
         * @brief Returns the dimensionality of the optimization problem.
         * @return Number of dimensions.
         */
        [[nodiscard]] virtual std::size_t size() const = 0;

        /**
         * @brief Short identifier used to select a GPU evaluation kernel.
         *
         * Override in benchmark subclasses to enable GPU-side evaluation
         * ("Sphere", "Rastrigin", "HeavyTrig"). Return empty string (default)
         * to use the CPU callback path.
         *
         * @return Problem name string, or "" for custom functions.
         */
        [[nodiscard]] virtual std::string name() const { return ""; }

        /**
         * @brief Gets the lower bound for a specific dimension.
         * @param dimension Index of the dimension.
         * @return Lower bound value.
         */
        virtual GeneType getLowerBound(std::size_t dimension) const = 0;

        /**
         * @brief Convenience method to get the lower bound of the first dimension.
         * @return Lower bound of dimension 0.
         */
		GeneType getLowerBound() const {
    		return getLowerBound(0); 
		}
        
        /**
         * @brief Gets the upper bound for a specific dimension.
         * @param dimension Index of the dimension.
         * @return Upper bound value.
         */
        virtual GeneType getUpperBound(std::size_t dimension) const = 0;

        /**
         * @brief Convenience method to get the upper bound of the first dimension.
         * @return Upper bound of dimension 0.
         */
        GeneType getUpperBound() const {
    		return getUpperBound(0);
		}
    };
}

#endif // FITNESS_FUNCTION_H
