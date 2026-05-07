#ifndef ALGORITHM_BUILDER_H
#define ALGORITHM_BUILDER_H

#pragma once

#include <memory>
#include <yaml-cpp/yaml.h>

#include "algorithms/Algorithm.h"
#include "algorithms/standard/StandardGA.h"
#include "algorithms/standard/StandardGAParams.h"
#include "algorithms/cellular/CellularGA.h"
#include "algorithms/cellular/CellularGAParams.h"
#include "algorithms/differential-evolution/DifferentialEvolutionGA.h"
#include "algorithms/differential-evolution/DEParams.h"
#ifdef GALIB_HAS_CUDA
#include "algorithms/StandardGACUDA.h"
#endif

#include "utils/OperatorBuilder.h"

#ifdef GALIB_HAS_MPI
#include "algorithms/island/IslandGA.h"
#include "algorithms/island/IslandGAParams.h"
#include "algorithms/island/migration/replacers/WorstReplacer.h"
#include "algorithms/island/migration/selectors/ElitismSelector.h"
#endif

#include "core/FitnessFunction.h"

#ifdef GALIB_HAS_MPI
#include "algorithms/island/topology/Topology.h"
#include "algorithms/island/topology/FullyConnectedTopology.h"
#include "algorithms/island/topology/OneWayRingTopology.h"
#include "algorithms/island/topology/BidirectionalRingTopology.h"
#include "algorithms/island/communication/serializers/BinarySerializer.h"
#include "algorithms/island/communication/communicators/MpiCommunicator.h"
#include "algorithms/island/communication/buffers/CircularBuffer.h"
#include <mpi.h>
#endif

namespace galib::utils {
    template <typename GeneType>
    class AlgorithmBuilder {
    private:
        struct Defaults {
            static constexpr double MUTATION_RATE = 0.05;
            static constexpr double CROSSOVER_RATE = 0.8;
            static constexpr std::size_t MAX_GENERATIONS = 100;
            static constexpr bool USE_ELITISM = true;
            static constexpr std::size_t THREADS = 1;

            struct Standard {
                static constexpr bool USE_CUDA = false;
            };

            struct Cellular {
                static constexpr std::size_t ROWS = 10;
                static constexpr std::size_t COLS = 10;
                static constexpr bool USE_LOCAL_ELITISM = true;
            };

            struct DE {
                static constexpr double F_WEIGHT = 0.8;
                static constexpr double CR_RATE = 0.9;
            };

            struct Island {
                static constexpr std::size_t MAX_GENERATIONS = 1000;
                static constexpr std::size_t MIGRATION_INTERVAL = 50;
                static constexpr std::size_t MIGRATION_SIZE = 5;
                static constexpr double IMMIGRATION_QUOTA = 0.1;
                static constexpr std::size_t BUFFER_CAPACITY = 10;
                static constexpr const char* TOPOLOGY = "fully_connected";
            };
        };

    public:
        /**
         * @brief Unified build method that determines the algorithm type from config.
         */
#ifdef GALIB_HAS_MPI
        static std::unique_ptr<Algorithm<GeneType>> build(
            const std::string& config_path,
            FitnessFunction<GeneType>& ff,
            MPI_Comm mpi_comm = MPI_COMM_NULL
        ) {
#else
            static std::unique_ptr<Algorithm<GeneType>> build(
                const std::string& config_path,
                FitnessFunction<GeneType>& ff
            ) {
#endif
            YAML::Node full_config = YAML::LoadFile(config_path);
            const auto node = full_config["algorithm"];
            if (!node) {
                throw std::invalid_argument("Missing 'algorithm' section in config: " + config_path);
            }

            const auto type = node["type"].as<std::string>("standard");
            std::unique_ptr<Algorithm<GeneType>> algo;

            if (type == "standard") {
                algo = buildStandardGA(full_config, ff);
            } else if (type == "cellular") {
                algo = buildCellularGA(full_config, ff);
            } else if (type == "differential_evolution") {
                algo = buildDifferentialEvolutionGA(full_config, ff);
            } else if (type == "island") {
#ifdef GALIB_HAS_MPI
                algo = buildIslandGA(full_config, ff, mpi_comm);
#else
                throw std::runtime_error("Island GA requested but library was built without MPI support.");
#endif
            } else {
                throw std::invalid_argument("Unknown algorithm type: " + type);
            }

            // Setup Logging
            if (full_config["output"]) {
                const auto out = full_config["output"];

                // Console Logging
                if (out["console"] && out["console"]["enabled"].as<bool>(false)) {
                    auto interval = out["console"]["interval"].as<std::size_t>(1);
                    algo->enableConsoleLogging(interval);
                }

                // File Logging
                if (out["file"] && out["file"]["enabled"].as<bool>(false)) {
                    auto path = out["file"]["path"].as<std::string>("evolution.csv");
                    auto interval = out["file"]["interval"].as<std::size_t>(1);
                    algo->enableFileLogging(path, interval);
                }
            }

            return algo;
        }

#ifdef GALIB_HAS_MPI
        static std::unique_ptr<Algorithm<GeneType>> build(
            const YAML::Node& config,
            FitnessFunction<GeneType>& ff,
            std::size_t threads = 0,
            MPI_Comm mpi_comm = MPI_COMM_WORLD
        ) {
#else
        static std::unique_ptr<Algorithm<GeneType>> build(
            const YAML::Node& config,
            FitnessFunction<GeneType>& ff,
            std::size_t threads = 0
        ) {
#endif
            const auto node = config["algorithm"];
            if (!node) {
                throw std::invalid_argument("Missing 'algorithm' section in config");
            }

            const auto type = node["type"].as<std::string>("standard");
            std::unique_ptr<Algorithm<GeneType>> algo;

            // Set threads in config if provided
            if (threads > 0) {
                const_cast<YAML::Node&>(config)["algorithm"]["threads"] = threads;
            }

            if (type == "standard") {
                algo = buildStandardGA(config, ff);
            } else if (type == "cellular") {
                algo = buildCellularGA(config, ff);
            } else if (type == "differential_evolution") {
                algo = buildDifferentialEvolutionGA(config, ff);
            } else if (type == "island") {
#ifdef GALIB_HAS_MPI
                algo = buildIslandGA(config, ff, mpi_comm);
#else
                throw std::runtime_error("Island GA requested but library was built without MPI support.");
#endif
            } else {
                throw std::invalid_argument("Unknown algorithm type: " + type);
            }

            // Setup Logging
            if (config["output"]) {
                const auto out = config["output"];

                // Console Logging
                if (out["console"] && out["console"]["enabled"].as<bool>(false)) {
                    auto interval = out["console"]["interval"].as<std::size_t>(1);
                    algo->enableConsoleLogging(interval);
                }

                // File Logging
                if (out["file"] && out["file"]["enabled"].as<bool>(false)) {
                    auto path = out["file"]["path"].as<std::string>("evolution.csv");
                    auto interval = out["file"]["interval"].as<std::size_t>(1);
                    algo->enableFileLogging(path, interval);
                }
            }

            return algo;
        }

        /**
         * @brief Builds a StandardGA instance from the root configuration.
         */
        static std::unique_ptr<Algorithm<GeneType>> buildStandardGA(
            const YAML::Node& config,
            FitnessFunction<GeneType>& ff
        ) {
            const auto node = config["algorithm"];
            StandardGAParams params;
            params.mutation_rate = node["mutation_rate"].as<double>(Defaults::MUTATION_RATE);
            params.crossover_rate = node["crossover_rate"].as<double>(Defaults::CROSSOVER_RATE);
            params.max_generations = node["max_generations"].as<std::size_t>(Defaults::MAX_GENERATIONS);
            params.use_elitism = node["use_elitism"].as<bool>(Defaults::USE_ELITISM);
            params.threads = node["threads"].as<std::size_t>(Defaults::THREADS);

            bool use_cuda = Defaults::Standard::USE_CUDA;
            if (node["standard"]) {
                use_cuda = node["standard"]["use_cuda"].as<bool>(Defaults::Standard::USE_CUDA);
            }

            auto selection = OperatorBuilder<GeneType>::buildSelection(node["selection"]);
            auto mutation = OperatorBuilder<GeneType>::buildMutation(node["mutation"], ff.getLowerBound(), ff.getUpperBound());
            auto crossover = OperatorBuilder<GeneType>::buildCrossover(node["crossover"]);

#ifdef GALIB_HAS_CUDA
            if (use_cuda) {
                // NOTE: StandardGACUDA must inherit from Algorithm<GeneType> 
                // and have a constructor compatible with StandardGA.
                return std::make_unique<cuda::StandardGACUDA<GeneType>>(
                    ff,
                    std::move(selection),
                    std::move(mutation),
                    std::move(crossover),
                    params.mutation_rate,
                    params.crossover_rate,
                    params.max_generations,
                    params.use_elitism
                );
            }
#endif

            return std::make_unique<StandardGA<GeneType>>(
                ff,
                std::move(selection),
                std::move(mutation),
                std::move(crossover),
                params.mutation_rate,
                params.crossover_rate,
                params.max_generations,
                params.use_elitism,
                params.threads
            );
        }

        /**
         * @brief Builds a CellularGA instance from the root configuration.
         */
        static std::unique_ptr<Algorithm<GeneType>> buildCellularGA(
            const YAML::Node& config,
            FitnessFunction<GeneType>& ff
        ) {
            const auto node = config["algorithm"];
            CellularGAParams params;
            params.mutation_rate = node["mutation_rate"].as<double>(Defaults::MUTATION_RATE);
            params.crossover_rate = node["crossover_rate"].as<double>(Defaults::CROSSOVER_RATE);
            params.max_generations = node["max_generations"].as<std::size_t>(Defaults::MAX_GENERATIONS);

            if (node["cellular"]) {
                params.use_local_elitism = node["cellular"]["use_local_elitism"].as<bool>(Defaults::Cellular::USE_LOCAL_ELITISM);
            }

            auto selection = OperatorBuilder<GeneType>::buildLocalSelection(node["selection"]);
            auto mutation = OperatorBuilder<GeneType>::buildMutation(node["mutation"], ff.getLowerBound(), ff.getUpperBound());
            auto crossover = OperatorBuilder<GeneType>::buildCrossover(node["crossover"]);

            return std::make_unique<CellularGA<GeneType>>(
                ff,
                std::move(selection),
                std::move(mutation),
                std::move(crossover),
                params.mutation_rate,
                params.crossover_rate,
                params.max_generations,
                params.use_local_elitism
            );
        }

        /**
         * @brief Builds a DifferentialEvolutionGA instance from the root configuration.
         */
        static std::unique_ptr<Algorithm<GeneType>> buildDifferentialEvolutionGA(
            const YAML::Node& config,
            FitnessFunction<GeneType>& ff
        ) {
            const auto node = config["algorithm"];
            DEParams params;
            params.crossover_rate = node["crossover_rate"].as<double>(Defaults::DE::CR_RATE);
            params.max_generations = node["max_generations"].as<std::size_t>(Defaults::MAX_GENERATIONS);
            params.threads = node["threads"].as<std::size_t>(Defaults::THREADS);

            if (node["differential_evolution"]) {
                params.f_weight = node["differential_evolution"]["f_weight"].as<double>(Defaults::DE::F_WEIGHT);
            }

            return std::make_unique<DifferentialEvolutionGA<GeneType>>(
                ff,
                params.f_weight,
                params.crossover_rate,
                params.max_generations,
                params.threads
            );
        }

        /**
         * @brief Builds an IslandGA instance from the root configuration.
         */
#ifdef GALIB_HAS_MPI
        static std::unique_ptr<Algorithm<GeneType>> buildIslandGA(
            const YAML::Node& config,
            FitnessFunction<GeneType>& ff,
            MPI_Comm mpi_comm
        ) {
            const auto node = config["algorithm"];
            IslandGAParams island_config;
            island_config.max_generations = node["max_generations"].as<std::size_t>(Defaults::Island::MAX_GENERATIONS);
            island_config.mutation_rate = node["mutation_rate"].as<double>(Defaults::MUTATION_RATE);
            island_config.crossover_rate = node["crossover_rate"].as<double>(Defaults::CROSSOVER_RATE);

            std::size_t buffer_capacity = Defaults::Island::BUFFER_CAPACITY;
            std::string topology_type = Defaults::Island::TOPOLOGY;

            if (node["island"]) {
                island_config.migration_interval = node["island"]["migration_interval"].as<std::size_t>(
                    Defaults::Island::MIGRATION_INTERVAL);
                island_config.migration_size = node["island"]["migration_size"].as<std::size_t>(Defaults::Island::MIGRATION_SIZE);
                island_config.immigration_quota = node["island"]["immigration_quota"].as<double>(Defaults::Island::IMMIGRATION_QUOTA);
                buffer_capacity = node["island"]["buffer_capacity"].as<std::size_t>(Defaults::Island::BUFFER_CAPACITY);
                topology_type = node["island"]["topology"].as<std::string>(Defaults::Island::TOPOLOGY);
            }

            auto selection = OperatorBuilder<GeneType>::buildSelection(node["selection"]);
            auto mutation = OperatorBuilder<GeneType>::buildMutation(node["mutation"], ff.getLowerBound(), ff.getUpperBound());
            auto crossover = OperatorBuilder<GeneType>::buildCrossover(node["crossover"]);

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

            auto serializer = std::make_unique<internal::BinarySerializer<GeneType>>();
            std::size_t max_payload = serializer->getSerializedSize(island_config.migration_size, ff.size());
            auto comm = std::make_unique<internal::MpiCommunicator<GeneType>>(*serializer, max_payload, mpi_comm);
            auto buffer = std::make_unique<internal::CircularBuffer<GeneType>>(buffer_capacity, island_config.migration_size);

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
#endif
    };
} // namespace galib::utils

#endif // ALGORITHM_BUILDER_H
