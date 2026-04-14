#include <iostream>
#include <mpi.h>

#include "algorithms/island/IslandGA.h"
#include "algorithms/island/IslandConfig.h"
#include "algorithms/island/communication/communicators/MpiCommunicator.h"
#include "algorithms/island/communication/serializers/BinarySerializer.h"
#include "algorithms/island/communication/buffers/CircularBuffer.h"
#include "algorithms/island/migration/replacers/WorstReplacer.h"
#include "algorithms/island/migration/selectors/ElitismSelector.h"
#include "algorithms/island/topology/FullyConnectedTopology.h"

#include "operators/selection/TournamentSelection.h"
#include "operators/crossover/SinglePointCrossover.h"
#include "operators/mutation/GaussianMutation.h"
#include "benchmarks/SphereFunction.h"
#include "core/Population.h"
#include "utils/StateLogger.h"

using namespace galib;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    try {
        constexpr std::size_t num_genes = 10;

        // 1. Basic GA Components
        benchmark::SphereFunction<double> fitness_fn(num_genes, -5.12, 5.12);
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
        config.log_interval = 5;
        config.log_directory = "island-logs";

        BinarySerializer<double> serializer;

        const std::size_t max_payload_size = serializer.getSerializedSize(config.migration_size, num_genes);

        MpiCommunicator<double> communicator(serializer, max_payload_size, MPI_COMM_WORLD);

        if (communicator.getSize() < 2) {
            if (communicator.getRank() == 0) {
                std::cerr << "Island Model requires at least 2 processes." << std::endl;
            }
            MPI_Finalize();
            return 1;
        }

        utils::StateLogger<double> logger(config.log_directory, communicator.getRank());
        CircularBuffer<double> buffer(config.buffer_capacity, config.migration_size);
        WorstReplacer<double> replacer;
        ElitismSelector<double> selector;
        const FullyConnectedTopology topology(communicator.getSize());

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
            config,
            true,
            &logger
        );

        // 4. Initialize Population
        Population<double> population(config.population_size, num_genes);
        population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

        if (communicator.getRank() == 0) {
            std::cout << "Starting Island Model GA with " << communicator.getSize() << " islands..." << std::endl;
        }

        island_ga.run(population);

        const auto& best = population.getBestIndividual();
        if (communicator.getRank() == 0) {
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
