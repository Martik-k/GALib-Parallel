#include <iostream>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include "benchmarks/SphereFunction.h"
#include "core/GridPopulation.h"
#include "utils/AlgorithmBuilder.h"

using namespace galib;

int main(int argc, char* argv[]) {
    try {
        std::string config_path = (argc > 1) ? argv[1] : "configs/full_config_example.yaml";
        YAML::Node full_config = YAML::LoadFile(config_path);

        const auto algorithm = full_config["algorithm"];
        if (!algorithm) {
            throw std::invalid_argument("Missing 'algorithm' section in config.");
        }

        const std::string algorithm_type = algorithm["type"].as<std::string>("cellular");
        if (algorithm_type != "cellular") {
            throw std::invalid_argument(
                "This example only supports algorithm.type = 'cellular'.");
        }

        const auto cellular_node = algorithm["cellular"];
        if (!cellular_node) {
            throw std::invalid_argument("Missing 'algorithm.cellular' section in config.");
        }

        constexpr std::size_t num_genes = 10;
        benchmark::SphereFunction<double> fitness_fn(num_genes, -5.12, 5.12);

        const std::size_t rows = cellular_node["rows"].as<std::size_t>(10);
        const std::size_t cols = cellular_node["cols"].as<std::size_t>(10);
        const std::string log_file =
            full_config["output"] ? full_config["output"]["log_file"].as<std::string>("") : "";

        GridPopulation<double> population(rows, cols, num_genes);
        population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

        const auto cellular_ga =
            utils::AlgorithmBuilder<double>::buildCellularGA(full_config, fitness_fn);

        if (!log_file.empty()) {
            cellular_ga->enableLogging(log_file);
        }

        std::cout << "Starting Cellular GA..." << std::endl;
        std::cout << "Configuration: " << config_path << std::endl;
        std::cout << "Grid: " << rows << "x" << cols << std::endl;

        cellular_ga->run(population);

        const auto& best = population.getBestIndividual();
        std::cout << "Optimization finished. Best Fitness: " << best.getFitness() << std::endl;
        std::cout << "Best Genotype: " << best << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
