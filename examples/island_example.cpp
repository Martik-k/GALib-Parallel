#include <iostream>
#include <vector>
#include <mpi.h>

#include "algorithms/island/IslandGA.h"
#include "algorithms/island/IslandConfig.h"
#include "algorithms/island/communication/communicators/MpiCommunicator.h"
#include "algorithms/island/communication/serializers/BinarySerializer.h"
#include "algorithms/island/communication/buffers/CircularBuffer.h"
#include "algorithms/island/migration/replacers/WorstReplacer.h"
#include "algorithms/island/migration/selectors/ElitismSelector.h"
#include "topology/OneWayRingTopology.h"

#include "operators/selection/TournamentSelection.h"
#include "operators/crossover/SinglePointCrossover.h"
#include "operators/mutation/GaussianMutation.h"
#include "benchmarks/SphereFunction.h"
#include "core/Population.h"

using namespace galib;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

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

    try {
        // 1. Basic GA Components
        benchmark::SphereFunction<double> fitness_fn(10, -5.12, 5.12); // 10 dimensions, standard bounds
        TournamentSelection<double> selection(3);
        GaussianMutation<double> mutation(0.05);
        SinglePointCrossover<double> crossover;

        // 2. Island Model Specific Components
        IslandConfig config;
        config.population_size = 50;
        config.max_generations = 200;
        config.migration_interval = 20;
        config.migration_size = 5;
        config.immigration_quota = 0.2;

        BinarySerializer<double> serializer;
        // Estimate buffer size: migration_size * (genes * sizeof(double) + sizeof(fitness)) + overhead
        const std::size_t receive_buffer_size = config.migration_size * (10 * sizeof(double) + sizeof(double)) + 100;
        MpiCommunicator<double> communicator(serializer, receive_buffer_size);
        
        CircularBuffer<double> buffer(config.buffer_capacity, config.migration_size);
        WorstReplacer<double> replacer;
        ElitismSelector<double> selector;
        OneWayRingTopology topology(size);

        // 3. Initialize Algorithm
        IslandGA<double> island_ga(
            fitness_fn,
            selection,
            mutation,
            crossover,
            replacer,
            selector,
            buffer,
            communicator,
            topology,
            config
        );

        // 4. Initialize Population using the library's built-in initialization
        Population<double> population(config.population_size, 10);
        population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

        if (communicator.getRank() == 0) {
            std::cout << "Starting Island Model GA with " << communicator.getSize() << " islands..." << std::endl;
        }

        island_ga.run(population);

        MPI_Barrier(MPI_COMM_WORLD);
        
        const auto& best = population.getBestIndividual();
        if (communicator.getRank() == 0) {
            std::cout << "Optimization finished. Global Best Fitness: " << best.getFitness() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " caught exception: " << e.what() << std::endl;
    }

    MPI_Finalize();
    return 0;
}
