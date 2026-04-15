#ifndef ALGORITHM_BUILDER_H
#define ALGORITHM_BUILDER_H

#pragma once

#include <memory>
#include <yaml-cpp/yaml.h>

#include "utils/OperatorBuilder.h"

#include "algorithms/standard/StandardGA.h"
#include "algorithms/standard/StandardGAParams.h"
#include "algorithms/cellular/CellularGA.h"
#include "algorithms/cellular/CellularGAParams.h"
#include "algorithms/differential-evolution/DifferentialEvolutionGA.h"
#include "algorithms/differential-evolution/DEParams.h"
#include "algorithms/island/IslandGA.h"
#include "algorithms/island/IslandConfig.h"

#include "core/FitnessFunction.h"

#include "algorithms/island/topology/Topology.h"
#include "algorithms/island/topology/FullyConnectedTopology.h"
#include "algorithms/island/topology/OneWayRingTopology.h"
#include "algorithms/island/topology/BidirectionalRingTopology.h"
#include "algorithms/island/communication/serializers/BinarySerializer.h"
#include "algorithms/island/communication/communicators/MpiCommunicator.h"
#include "algorithms/island/communication/buffers/CircularBuffer.h"
#include <mpi.h>

namespace galib::utils {

    template <typename GeneType>
    class AlgorithmBuilder {
    public:
        /**
         * @brief Builds a StandardGA instance from the root configuration.
         */
        static std::unique_ptr<StandardGA<GeneType>> buildStandardGA(
            const YAML::Node& config, 
            FitnessFunction<GeneType>& ff
        ) {
            const auto node = config["algorithm"];
            StandardGAParams params;
            params.mutation_rate = node["mutation_rate"].as<double>(0.05);
            params.crossover_rate = node["crossover_rate"].as<double>(0.8);
            params.max_generations = node["max_generations"].as<std::size_t>(100);
            params.use_elitism = node["use_elitism"].as<bool>(true);

            bool use_cuda = false;
            if (node["standard"]) {
                use_cuda = node["standard"]["use_cuda"].as<bool>(false);
            }

            auto selection = OperatorBuilder<GeneType>::buildSelection(node["selection"]);
            auto mutation = OperatorBuilder<GeneType>::buildMutation(node["mutation"], ff.getLowerBound(), ff.getUpperBound());
            auto crossover = OperatorBuilder<GeneType>::buildCrossover(node["crossover"]);

            return nullptr; // Placeholder for now
        }

        /**
         * @brief Builds a CellularGA instance from the root configuration.
         */
        static std::unique_ptr<CellularGA<GeneType>> buildCellularGA(
            const YAML::Node& config, 
            FitnessFunction<GeneType>& ff
        ) {
            const auto node = config["algorithm"];
            CellularGAParams params;
            params.mutation_rate = node["mutation_rate"].as<double>(0.05);
            params.crossover_rate = node["crossover_rate"].as<double>(0.8);
            params.max_generations = node["max_generations"].as<std::size_t>(100);
            
            if (node["cellular"]) {
                params.rows = node["cellular"]["rows"].as<std::size_t>(10);
                params.cols = node["cellular"]["cols"].as<std::size_t>(10);
                params.use_local_elitism = node["cellular"]["use_local_elitism"].as<bool>(true);
            }

            auto selection = OperatorBuilder<GeneType>::buildLocalSelection(node["selection"]);
            auto mutation = OperatorBuilder<GeneType>::buildMutation(node["mutation"], ff.getLowerBound(), ff.getUpperBound());
            auto crossover = OperatorBuilder<GeneType>::buildCrossover(node["crossover"]);

            return nullptr; 
        }

        /**
         * @brief Builds a DifferentialEvolutionGA instance from the root configuration.
         */
        static std::unique_ptr<DifferentialEvolutionGA<GeneType>> buildDifferentialEvolutionGA(
            const YAML::Node& config,
            FitnessFunction<GeneType>& ff
        ) {
            const auto node = config["algorithm"];
            DEParams params;
            params.cr_rate = node["crossover_rate"].as<double>(0.9);
            params.max_generations = node["max_generations"].as<std::size_t>(100);

            if (node["differential_evolution"]) {
                params.f_weight = node["differential_evolution"]["f_weight"].as<double>(0.8);
            }

            return nullptr; 
        }

        /**
         * @brief Builds an IslandGA instance from the root configuration.
         * This handles the creation of the Communicator, Serializer, Buffer, and Topology.
         */
        static std::unique_ptr<IslandGA<GeneType>> buildIslandGA(
            const YAML::Node& config, 
            FitnessFunction<GeneType>& ff,
            MPI_Comm mpi_comm
        ) {
            const auto node = config["algorithm"];
            IslandConfig island_config;
            island_config.max_generations = node["max_generations"].as<std::size_t>(1000);
            island_config.mutation_rate = node["mutation_rate"].as<double>(0.05);
            island_config.crossover_rate = node["crossover_rate"].as<double>(0.8);

            std::size_t buffer_capacity = 10;
            std::string topology_type = "fully_connected";

            if (node["island"]) {
                island_config.migration_interval = node["island"]["migration_interval"].as<std::size_t>(50);
                island_config.migration_size = node["island"]["migration_size"].as<std::size_t>(5);
                island_config.immigration_quota = node["island"]["immigration_quota"].as<double>(0.1);
                buffer_capacity = node["island"]["buffer_capacity"].as<std::size_t>(10);
                topology_type = node["island"]["topology"].as<std::string>("fully_connected");
            }

            // 1. Setup Genetic Operators
            auto selection = OperatorBuilder<GeneType>::buildSelection(node["selection"]);
            auto mutation = OperatorBuilder<GeneType>::buildMutation(node["mutation"], ff.getLowerBound(), ff.getUpperBound());
            auto crossover = OperatorBuilder<GeneType>::buildCrossover(node["crossover"]);
            
            // 2. Setup Migration Operators
            std::unique_ptr<DemeReplacer<GeneType>> replacer;
            if (node["island"] && node["island"]["replacer"]) {
                replacer = OperatorBuilder<GeneType>::buildDemeReplacer(node["island"]["replacer"]);
            } else {
                replacer = std::make_unique<WorstReplacer<GeneType>>();
            }

            std::unique_ptr<DemeSelector<GeneType>> selector;
            if (node["island"] && node["island"]["selector"]) {
                selector = OperatorBuilder<GeneType>::buildDemeSelector(node["island"]["selector"]);
            } else {
                selector = std::make_unique<ElitismSelector<GeneType>>();
            }

            // 3. Setup Infrastructure
            auto serializer = std::make_unique<BinarySerializer<GeneType>>();
            std::size_t max_payload = serializer->getSerializedSize(island_config.migration_size, ff.size());
            auto comm = std::make_unique<MpiCommunicator<GeneType>>(*serializer, max_payload, mpi_comm);
            auto buffer = std::make_unique<CircularBuffer<GeneType>>(buffer_capacity, island_config.migration_size);
            
            std::unique_ptr<const Topology> topology;
            int world_size;
            MPI_Comm_size(mpi_comm, &world_size);
            
            if (topology_type == "fully_connected") {
                topology = std::make_unique<FullyConnectedTopology>(world_size);
            } else if (topology_type == "one_way_ring") {
                topology = std::make_unique<OneWayRingTopology>(world_size);
            } else if (topology_type == "bidirectional_ring") {
                topology = std::make_unique<BidirectionalRingTopology>(world_size);
            } else {
                throw std::invalid_argument("Unknown topology type: " + topology_type);
            }

            bool elitism = node["use_elitism"].as<bool>(true);

            return std::make_unique<IslandGA<GeneType>>(
                ff, 
                std::move(selection), 
                std::move(mutation), 
                std::move(crossover), 
                std::move(replacer), 
                std::move(selector), 
                std::move(buffer), 
                std::move(comm), 
                std::move(topology), 
                std::move(serializer),
                island_config, 
                elitism
            );
        }
    };

} // namespace galib::utils

#endif // ALGORITHM_BUILDER_H
