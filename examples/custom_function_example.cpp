#include "algorithms/standard/StandardGA.h"
#include "utils/FunctionalFitness.h"
#include "operators/selection/TournamentSelection.h"
#include "operators/crossover/ArithmeticCrossover.h"
#include "operators/mutation/GaussianMutation.h"
#include <iostream>
#include <cmath>

int main() {
    // 1. Define a custom problem using a lambda
    // f(x, y) = (x-3)^2 + (y+2)^2 + 10
    // Global minimum is at (3, -2) with value 10
    auto custom_fitness = std::make_unique<galib::FunctionalFitness<double>>(
        2,      // 2 dimensions
        -10.0,  // lower bound
        10.0,   // upper bound
        [](const std::vector<double>& genes) {
            double x = genes[0];
            double y = genes[1];
            return std::pow(x - 3.0, 2) + std::pow(y + 2.0, 2) + 10.0;
        }
    );

    // 2. Configure GA parameters
    // galib::StandardGAParams params;
    // params.pop_size = 100;
    // params.max_generations = 100;
    // params.mutation_rate = 0.1;
    // params.crossover_rate = 0.8;
    //
    // // 3. Setup operators
    // auto selection = std::make_unique<galib::TournamentSelection<double>>(3);
    // auto crossover = std::make_unique<galib::ArithmeticCrossover<double>>();
    // auto mutation = std::make_unique<galib::GaussianMutation<double>>(0.0, 0.5);
    //
    // // 4. Create and run GA
    // galib::StandardGA<double> ga(
    //     *custom_fitness,
    //     std::move(selection),
    //     std::move(crossover),
    //     std::move(mutation),
    //     params
    // );
    //
    // galib::Population<double> population(params.pop_size, custom_fitness->size());
    // population.initialize(custom_fitness->getLowerBound(), custom_fitness->getUpperBound());
    //
    // ga.run(population);
    //
    // // 5. Output results
    // const auto& best = population.getBestIndividual();
    // std::cout << "\nOptimization Results:" << std::endl;
    // std::cout << "Best Fitness: " << best.getFitness() << std::endl;
    // std::cout << "Best Solution: x=" << best.getGenotype()[0] << ", y=" << best.getGenotype()[1] << std::endl;
    // std::cout << "Expected: x=3, y=-2, fitness=10" << std::endl;

    return 0;
}
