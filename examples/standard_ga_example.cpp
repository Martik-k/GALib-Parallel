#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>

#include "utils/Config.h"
#include "utils/ConfigParser.h"
#include "utils/FitnessFactory.h"

#include "core/Individual.h"
#include "core/Population.h"
#include "core/FitnessFunction.h"
#include "algorithms/standard/StandardGA.h"
#include "algorithms/StandardGACUDA.h"
#include "algorithms/differential-evolution/DifferentialEvolutionGA.h"

#include "operators/selection/TournamentSelection.h"
#include "operators/crossover/SinglePointCrossover.h"
#include "operators/mutation/GaussianMutation.h"

using namespace galib;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_config.yaml>" << std::endl;
        std::cerr << "Example: ./ga_example ../configs/config_file.yaml" << std::endl;
        return 1;
    }

    try {
        std::string config_path = argv[1];
        const utils::Config config = utils::ConfigParser::parse(config_path);

        std::cout << "=== GALib-Parallel: Optimization Experiment ===" << std::endl;
        std::cout << "Config file:     " << config_path << std::endl;
        std::cout << "Function:        " << config.problem.name << " (" << config.problem.dimensions << "D)" << std::endl;
        std::cout << "Population size: " << config.algorithm.pop_size << std::endl;
        std::cout << "Generations:     " << config.algorithm.max_generations << std::endl;
        std::cout << "Mutation rate:   " << config.algorithm.mutation_rate << std::endl;
        std::cout << "Crossover rate:  " << config.algorithm.crossover_rate << std::endl;
        std::cout << "Backend:         " << config.algorithm.backend << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;

        auto fitness_fn = utils::FitnessFactory::create(config.problem);
        Population<double> population(config.algorithm.pop_size, config.problem.dimensions);

        const bool use_cuda =
            (config.algorithm.backend == "CUDA" || config.algorithm.backend == "cuda");

        if (use_cuda) {
#if defined(GALIB_WITH_CUDA)
            std::cout << "Starting evolution on CUDA backend...\n" << std::endl;

            const cuda::StandardGACUDAConfig cuda_config =
                cuda::StandardGACUDA::fromConfig(config);
            const cuda::StandardGACUDA cuda_ga(cuda_config, *fitness_fn);

            if (!cuda_ga.run(population)) {
                throw std::runtime_error("CUDA GA execution failed.");
            }
#else
            throw std::runtime_error(
                "Config requested CUDA backend, but project was built without CUDA support.");
#endif
        } else {
            population.initialize(fitness_fn->getLowerBound(), fitness_fn->getUpperBound());

            if (config.algorithm.backend == "DE" || config.algorithm.backend == "OpenMP_DE") {
                // Differential Evolution GA
                DifferentialEvolutionGA<double> de(
                    *fitness_fn,
                    config.algorithm.mutation_rate,  // F
                    config.algorithm.crossover_rate, // CR
                    config.algorithm.max_generations
                );

                std::cout << "Starting Differential Evolution (Hybrid OpenMP+SIMD)...\n" << std::endl;

                de.enableLogging(config.output.log_file);
                de.run(population);

            } else {
                // Standard GA
                TournamentSelection<double> selector(config.algorithm.selection.tournament_size);
                SinglePointCrossover<double> crossover;
                GaussianMutation<double> mutation(config.algorithm.mutation_rate);

                StandardGA<double> ga(
                    *fitness_fn,
                    selector,
                    mutation,
                    crossover,
                    config.algorithm.mutation_rate,
                    config.algorithm.crossover_rate,
                    config.algorithm.max_generations,
                    true
                );

                std::cout << "Starting Standard GA (OpenMP active if compiled)...\n" << std::endl;

                ga.enableLogging(config.output.log_file);
                ga.run(population);
            }
        }

        std::cout << "\n=== Evolution Finished ===" << std::endl;

        const auto& best_individual = population.getBestIndividual();

        std::cout << "Best Fitness Found: " << std::fixed
                  << best_individual.getFitness() << " (Perfect is 0.0)" << std::endl;

        std::cout << "Best Genotype:      " << best_individual << std::endl;
    }

    catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}