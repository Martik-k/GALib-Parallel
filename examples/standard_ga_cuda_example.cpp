#include <iostream>
#include <string>

#include <yaml-cpp/yaml.h>

#include "benchmarks/RastriginFunction.h"
#include "core/Population.h"
#include "utils/AlgorithmBuilder.h"

using namespace galib;

int main(int argc, char* argv[]) {
    try {
        std::string config_path = (argc > 1) ? argv[1] : "configs/full_config_example.yaml";

        constexpr std::size_t NUM_GENES = 50;
        const std::size_t POPULATION_SIZE = 1000;

        benchmark::RastriginFunction<double> fitness_fn(NUM_GENES, -5.12, 5.12);

        const auto algo = utils::AlgorithmBuilder<double>::build(config_path, fitness_fn);

        Population<double> population(POPULATION_SIZE, NUM_GENES);
        population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

        // Determine actual backend from config at runtime
        const YAML::Node cfg      = YAML::LoadFile(config_path);
        const YAML::Node std_node = cfg["algorithm"]["standard"];
        const bool use_cuda       = std_node && std_node["use_cuda"] ? std_node["use_cuda"].as<bool>(false) : false;

#ifdef GALIB_WITH_CUDA
        if (use_cuda)
            std::cout << "Backend: StandardGACUDA (CUDA build, use_cuda=true)" << std::endl;
        else
            std::cout << "Backend: StandardGA    (CUDA build, use_cuda=false)" << std::endl;
#else
        std::cout << "Backend: StandardGA (CPU-only build, CUDA not compiled in)" << std::endl;
#endif
        std::cout << "Configuration: " << config_path << std::endl;
        std::cout << "Population size: " << POPULATION_SIZE << std::endl;

        algo->enableConsoleLogging(50);
        // algo->enableFileLogging("logs/evolution.csv", 1);  // Commented out, as build() already sets up logging from YAML

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
