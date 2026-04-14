#include <iostream>
#include <mpi.h>
#include <yaml-cpp/yaml.h>

#include "utils/AlgorithmBuilder.h"
#include "benchmarks/SphereFunction.h"
#include "core/Population.h"

using namespace galib;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    try {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (size < 2) {
            if (rank == 0) {
                std::cerr << "Island Model requires at least 2 processes." << std::endl;
            }
            MPI_Finalize();
            return 1;
        }

        std::string config_path = (argc > 1) ? argv[1] : "configs/full_config_example.yaml";
        YAML::Node full_config = YAML::LoadFile(config_path);

        constexpr std::size_t num_genes = 10;
        
        benchmark::SphereFunction<double> fitness_fn(num_genes, -5.12, 5.12);

        const auto island_ga = utils::AlgorithmBuilder<double>::buildIslandGA(
            full_config, 
            fitness_fn, 
            MPI_COMM_WORLD
        );

        island_ga->enableConsoleOutput(true);
        island_ga->enableFileLogging("logs/island_evolution", 10);

        std::size_t pop_size = full_config["algorithm"]["pop_size"].as<std::size_t>(100);
        Population<double> population(pop_size, num_genes);
        population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

        if (rank == 0) {
            std::cout << "Starting Island Model GA with " << size << " islands..." << std::endl;
            std::cout << "Configuration: " << config_path << std::endl;
        }

        island_ga->run(population);

        if (rank == 0) {
            const auto& best = population.getBestIndividual();
            std::cout << "Optimization finished. Global Best Fitness: " << best.getFitness() << std::endl;
        }

    } catch (const std::exception& e) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cerr << "Rank " << rank << " caught exception: " << e.what() << std::endl;
    }

    MPI_Finalize();
    return 0;
}
