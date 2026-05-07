#include "utils/AlgorithmBuilder.h"
#include "utils/FunctionalFitness.h"
#include <iostream>

using namespace galib;

int main() {
    // 1. Define problem
    FunctionalFitness<double> fitness(2, -5.0, 5.0,
        [](const std::vector<double>& p) {
            return p[0] * p[0] + p[1] * p[1];
        }
    );

    // 2. Build algorithm instance from YAML configuration
    auto ga = utils::AlgorithmBuilder<double>::build(
        "configs/minimal_config.yaml", fitness
    );

    // 3. Evolve the population
    Population<double> pop(100, 2);
    pop.initialize(
        fitness.getLowerBound(),
        fitness.getUpperBound()
    );
    ga->run(pop);

    // 4. Get results
    std::cout
        << "Best Fitness: "
        << pop.getBestIndividual().getFitness()
        << std::endl;

    return 0;
}
