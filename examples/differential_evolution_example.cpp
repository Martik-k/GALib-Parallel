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
        std::string config_path = (argc > 1) ? argv[1] : "configs/config_de.yaml";
        YAML::Node full_config = YAML::LoadFile(config_path);

        const auto algorithm = full_config["algorithm"];
        if (!algorithm) {
            throw std::invalid_argument("Missing 'algorithm' section in config.");
        }

        const std::string algorithm_type = algorithm["type"].as<std::string>("differential_evolution");
        if (algorithm_type != "differential_evolution") {
            throw std::invalid_argument(
                "This example only supports algorithm.type = 'differential_evolution'.");
        }

        constexpr std::size_t num_genes = 10;
        benchmark::RastriginFunction<double> fitness_fn(num_genes, -5.12, 5.12);

        const std::size_t pop_size = algorithm["pop_size"].as<std::size_t>(50);
        const std::string log_file =
            full_config["output"] ? full_config["output"]["log_file"].as<std::string>("") : "";

        Population<double> population(pop_size, num_genes);
        population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

        const auto de_ga =
            utils::AlgorithmBuilder<double>::buildDifferentialEvolutionGA(full_config, fitness_fn);

        if (!log_file.empty()) {
            de_ga->enableLogging(log_file);
        }

        std::cout << "Starting Differential Evolution GA..." << std::endl;
        std::cout << "Configuration: " << config_path << std::endl;
        std::cout << "Population size: " << pop_size << std::endl;

        de_ga->run(population);

        const auto& best = population.getBestIndividual();
        std::cout << "Optimization finished. Best Fitness: " << best.getFitness() << std::endl;
        std::cout << "Best Genotype: " << best << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}