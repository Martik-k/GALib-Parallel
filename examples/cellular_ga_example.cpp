#include <iostream>
#include <stdexcept>
#include <string>

#include "benchmarks/SphereFunction.h"
#include "core/Population.h"
#include "utils/AlgorithmBuilder.h"

using namespace galib;

int main(int argc, char* argv[]) {
    try {
        std::string config_path = (argc > 1) ? argv[1] : "configs/full_config_example.yaml";

        constexpr std::size_t num_genes = 10;
        constexpr std::size_t population_size = 100;
        benchmark::SphereFunction<double> fitness_fn(num_genes, -5.12, 5.12);

        Population<double> population(population_size, num_genes);
        population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

        const auto cellular_ga = utils::AlgorithmBuilder<double>::build(config_path, fitness_fn);

        std::cout << "Starting Cellular GA..." << std::endl;
        std::cout << "Configuration: " << config_path << std::endl;
        std::cout << "Population size: " << population_size << std::endl;

        cellular_ga->enableConsoleLogging(10);
        cellular_ga->enableFileLogging("logs/cellular_ga", 1);

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
