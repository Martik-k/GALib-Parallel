#include <iostream>
#include <mpi.h>
#include <yaml-cpp/yaml.h>

#include "utils/AlgorithmBuilder.h"
#include "benchmarks/RastriginFunction.h"
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

        std::string config_path = (argc > 1) ? argv[1] : "configs/config_island.yaml";
        YAML::Node full_config = YAML::LoadFile(config_path);

        constexpr std::size_t NUM_GENES = 2;
        constexpr std::size_t POPULATION_SIZE = 30;

        benchmark::RastriginFunction<double> fitness_fn(NUM_GENES, -5.12, 5.12);

        const auto island_ga = utils::AlgorithmBuilder<double>::buildIslandGA(
            full_config, 
            fitness_fn, 
            MPI_COMM_WORLD
        );

        island_ga->enableConsoleOutput(true, 25);
        island_ga->enableFileLogging("logs/island_evolution", 1);

        Population<double> population(POPULATION_SIZE, NUM_GENES);
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
