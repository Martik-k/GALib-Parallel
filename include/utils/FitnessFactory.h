#pragma once

#include "utils/Config.h"
#include "core/FitnessFunction.h"
#include "benchmarks/SphereFunction.h"
#include "benchmarks/RastriginFunction.h"
#include "benchmarks/HimmelblauFunction.h"
#include "benchmarks/DeJongF5Function.h"
#include "benchmarks/HeavyTrigFunction.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace galib::utils {
    /**
     * @brief A factory for instantiating standard benchmark functions.
     * 
     * Maps string names (e.g., "Sphere", "Rastrigin") to concrete 
     * FitnessFunction implementations.
     */
    class FitnessFactory {
    public:
        /**
         * @brief Creates a fitness function based on the provided configuration.
         * @param problem Configuration containing function name and bounds.
         * @return unique_ptr to the constructed FitnessFunction.
         * @throws std::invalid_argument if the function name is unrecognized.
         */
        static std::unique_ptr<FitnessFunction<double>> create(const ProblemConfig& problem) {
            const std::string& name = problem.name;
            const std::size_t dim = problem.dimensions;
            const double lb = problem.lower_bound;
            const double ub = problem.upper_bound;

            if (name == "Sphere") return std::make_unique<benchmark::SphereFunction<double>>(dim, lb, ub);
            if (name == "Rastrigin") return std::make_unique<benchmark::RastriginFunction<double>>(dim, lb, ub);
            if (name == "Himmelblau") return std::make_unique<benchmark::HimmelblauFitness<double>>();
            if (name == "DeJongF5") return std::make_unique<benchmark::DeJongF5Function<double>>();
            if (name == "HeavyTrig") return std::make_unique<benchmark::HeavyTrigFunction<double>>(dim, lb, ub);

            throw std::invalid_argument("Unknown fitness function: '" + name + "'");
        }
    };
} // namespace galib::utils
