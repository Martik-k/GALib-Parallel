#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

#include "core/Individual.h"
#include "core/Population.h"
#include "core/FitnessFunction.h"
#include "algorithms/StandardGA.h"

#include "operators/selection/TournamentSelection.h"
#include "operators/crossover/SinglePointCrossover.h"
#include "operators/mutation/GaussianMutation.h"

#include "benchmarks/RastriginFunction.h"
#include "benchmarks/SphereFunction.h"

using namespace galib;

void printGenotype(const Individual<double>& ind) {
    std::cout << "[ ";
    std::size_t print_limit = std::min(ind.getGenotype().size(), std::size_t(5));
    for (std::size_t i = 0; i < print_limit; ++i) {
        std::cout << std::fixed << std::setprecision(4) << std::setw(8) << ind.getGenotype()[i] << " ";
    }
    if (ind.getGenotype().size() > 5) std::cout << "... ";
    std::cout << "]";
}

int main() {
    const std::size_t num_genes = 5;
    const std::size_t pop_size = 1000;
    const std::size_t max_generations = 100;

    std::cout << "=== GALib-Parallel: 10D Rastrigin Optimization ===" << std::endl;
    std::cout << "Dimensions:      " << num_genes << std::endl;
    std::cout << "Population size: " << pop_size << std::endl;
    std::cout << "Generations:     " << max_generations << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    galib::benchmark::SphereFunction<double> fitness_fn(5);

    TournamentSelection<double> selector(3);
    SinglePointCrossover<double> crossover;
    GaussianMutation<double> mutation(0.2);

    Population<double> population(pop_size, num_genes);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

    for (auto& individual : population) {
        for (double& gene : individual.getGenotype()) {
            gene = dis(gen);
        }
    }

    StandardGA<double> ga(
        fitness_fn,
        selector,
        mutation,
        crossover,
        0.2,
        0.9,
        max_generations,
        true
    );

    std::cout << "Starting evolution (OpenMP active)...\n" << std::endl;
	ga.enableLogging("evolution_history.csv");
    ga.run(population);

    std::cout << "\n=== Evolution Finished ===" << std::endl;
    const auto& best_individual = population.getBestIndividual();

    std::cout << "Best Fitness Found: " << std::fixed << std::setprecision(6)
              << best_individual.getFitness() << " (Perfect is 0.0)" << std::endl;

    std::cout << "Best Genotype:      ";
    printGenotype(best_individual);
    std::cout << std::endl;

    return 0;
}