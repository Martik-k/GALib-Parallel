#include <iostream>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include "benchmarks/RastriginFunction.h"
#include "core/Population.h"
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

        const std::string algorithm_type = algorithm["type"].as<std::string>("standard");
        if (algorithm_type != "standard") {
            throw std::invalid_argument(
                "This example only supports algorithm.type = 'standard'.");
        }

        constexpr std::size_t num_genes = 10;
        benchmark::RastriginFunction<double> fitness_fn(num_genes, -5.12, 5.12);

        const std::size_t pop_size = algorithm["pop_size"].as<std::size_t>(50);

        const auto algo = utils::AlgorithmBuilder<double>::build(config_path, fitness_fn);

        Population<double> population(pop_size, num_genes);
        population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

        std::cout << "Starting Standard GA..." << std::endl;
        std::cout << "Configuration: " << config_path << std::endl;
        std::cout << "Population size: " << pop_size << std::endl;

        algo->enableConsoleLogging(50);
        algo->enableFileLogging("logs/evolution.csv", 1);

        algo->run(population);

        const auto& best = population.getBestIndividual();
        std::cout << "Optimization finished. Best Fitness: " << best.getFitness() << std::endl;
        std::cout << "Best Genotype: " << best << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}