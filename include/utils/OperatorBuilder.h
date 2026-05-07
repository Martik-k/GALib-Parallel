#ifndef OPERATOR_BUILDER_H
#define OPERATOR_BUILDER_H

#pragma once

#include <memory>
#include <string>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

#include "operators/selection/Selection.h"
#include "operators/selection/TournamentSelection.h"
#include "algorithms/cellular/selection/LocalSelection.h"
#include "algorithms/cellular/selection/BestNeighborSelection.h"
#include "operators/mutation/Mutation.h"
#include "operators/mutation/GaussianMutation.h"
#include "operators/mutation/UniformMutation.h"
#include "operators/mutation/BoundaryMutation.h"

#include "operators/crossover/Crossover.h"
#include "operators/crossover/ArithmeticCrossover.h"
#include "operators/crossover/SinglePointCrossover.h"
#include "operators/crossover/UniformCrossover.h"

#include "algorithms/cellular/selection/LocalSelection.h"
#include "algorithms/cellular/selection/BestNeighborSelection.h"

#include "algorithms/island/migration/replacers/DemeReplacer.h"
#include "algorithms/island/migration/replacers/WorstReplacer.h"
#include "algorithms/island/migration/selectors/DemeSelector.h"
#include "algorithms/island/migration/selectors/ElitismSelector.h"

namespace galib::internal {
    /**
     * @brief Internal factory for instantiating GA operators from YAML configurations.
     * 
     * This class is primarily used by AlgorithmBuilder to map configuration 
     * strings to concrete operator instances.
     * 
     * @tparam GeneType The numeric type of each gene.
     */
    template <typename GeneType = double>
    class OperatorBuilder {
    public:
        /**
         * @brief Builds a selection operator.
         * @param node YAML node for selection config.
         * @return unique_ptr to Selection.
         */
        static std::unique_ptr<Selection<GeneType>> buildSelection(const YAML::Node& node) {
            const auto type = node["type"].as<std::string>("tournament");

            if (type == "tournament") {
                const auto size = node["tournament_size"].as<std::size_t>(3);
                return std::make_unique<TournamentSelection<GeneType>>(size);
            }

            throw std::invalid_argument("Unknown Selection type: " + type);
        }

        /**
         * @brief Builds a local selection operator (for Cellular GA).
         * @param node YAML node for selection config.
         * @return unique_ptr to LocalSelection.
         */
        static std::unique_ptr<LocalSelection<GeneType>> buildLocalSelection(const YAML::Node& node) {
            const auto type = node["type"].as<std::string>("best_neighbor");

            if (type == "best_neighbor" || type == "tournament") {
                return std::make_unique<BestNeighborSelection<GeneType>>();
            }

            throw std::invalid_argument("Unknown LocalSelection type: " + type);
        }

        /**
         * @brief Builds a mutation operator.
         * @param node YAML node for mutation config.
         * @param lb   Lower bound (for uniform/boundary mutation).
         * @param ub   Upper bound (for uniform/boundary mutation).
         * @return unique_ptr to Mutation.
         */
        static std::unique_ptr<Mutation<GeneType>> buildMutation(const YAML::Node& node, GeneType lb = 0, GeneType ub = 0) {
            const auto type = node["type"].as<std::string>("gaussian");

            if (type == "gaussian") {
                auto sigma = node["sigma"].as<double>(0.1);
                return std::make_unique<GaussianMutation<GeneType>>(sigma);
            } else if (type == "uniform") {
                return std::make_unique<UniformMutation<GeneType>>(lb, ub);
            } else if (type == "boundary") {
                return std::make_unique<BoundaryMutation<GeneType>>(lb, ub);
            }

            throw std::invalid_argument("Unknown Mutation type: " + type);
        }

        /**
         * @brief Builds a crossover operator.
         * @param node YAML node for crossover config.
         * @return unique_ptr to Crossover.
         */
        static std::unique_ptr<Crossover<GeneType>> buildCrossover(const YAML::Node& node) {
            const auto type = node["type"].as<std::string>("single_point");

            if (type == "single_point") {
                return std::make_unique<SinglePointCrossover<GeneType>>();
            } else if (type == "arithmetic") {
                return std::make_unique<ArithmeticCrossover<GeneType>>();
            } else if (type == "uniform") {
                return std::make_unique<UniformCrossover<GeneType>>();
            }

            throw std::invalid_argument("Unknown Crossover type: " + type);
        }

        /**
         * @brief Builds a deme replacer policy (for Island GA).
         * @param node YAML node for replacer config.
         * @return unique_ptr to DemeReplacer.
         */
        static std::unique_ptr<DemeReplacer<GeneType>> buildDemeReplacer(const YAML::Node& node) {
            const auto type = node["type"].as<std::string>("worst");

            if (type == "worst") {
                return std::make_unique<WorstReplacer<GeneType>>();
            }

            throw std::invalid_argument("Unknown DemeReplacer type: " + type);
        }

        /**
         * @brief Builds a deme selector policy (for Island GA).
         * @param node YAML node for selector config.
         * @return unique_ptr to DemeSelector.
         */
        static std::unique_ptr<DemeSelector<GeneType>> buildDemeSelector(const YAML::Node& node) {
            const auto type = node["type"].as<std::string>("elitism");

            if (type == "elitism") {
                return std::make_unique<ElitismSelector<GeneType>>();
            }

            throw std::invalid_argument("Unknown DemeSelector type: " + type);
        }
    };
} // namespace galib::internal

#endif // OPERATOR_BUILDER_H
