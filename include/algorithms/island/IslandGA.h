#ifndef ISLAND_GA_H
#define ISLAND_GA_H

#pragma once

#include "core/Population.h"
#include "core/FitnessFunction.h"
#include "operators/selection/Selection.h"
#include "operators/mutation/Mutation.h"
#include "operators/crossover/Crossover.h"

#include "algorithms/island/migration/replacers/DemeReplacer.h"
#include "algorithms/island/migration/selectors/DemeSelector.h"
#include "communication/buffers/MigrationBuffer.h"
#include "communication/communicators/Communicator.h"
#include "algorithms/island/IslandConfig.h"
#include "topology/Topology.h"

#include <iostream>
#include <random>
#include <utility>
#include <algorithm>


namespace galib {
    template <typename GeneType = double>
    class IslandGA {
    private:
        FitnessFunction<GeneType>& fitness_function_m;
        Selection<GeneType>& selection_m;
        Mutation<GeneType>& mutation_m;
        Crossover<GeneType>& crossover_m;
        DemeReplacer<GeneType>& deme_replacer_m;
        DemeSelector<GeneType>& deme_selector_m;
        MigrationBuffer<GeneType>& migration_buffer_m;
        Communicator<GeneType>& communicator_m;
        const Topology& topology_m;

        const IslandConfig& config_m;

        bool use_elitism_m;

        const std::size_t max_immigrants_m;

        void evaluatePopulation(Population<GeneType>& population) {
            const std::size_t population_size = population.size();

            #pragma omp parallel for schedule(dynamic)
            for (std::size_t i = 0; i < population_size; ++i) {
                double score = fitness_function_m.evaluate(population[i].getGenotype());
                population[i].setFitness(score);
            }
        }

        void generateNextGeneration(const Population<GeneType>& current_population, Population<GeneType>& new_population) {
            const std::size_t population_size = current_population.size();
            const std::size_t elitism_offset = 0;

            if (use_elitism_m) {
                new_population[0] = current_population.getBestIndividual();
                elitism_offset = 1;
            }

            #pragma omp parallel for schedule(static)
            for (std::size_t i = elitism_offset; i < population_size; i += 2) {
                thread_local static std::random_device tl_rd;
                thread_local static std::mt19937_64 tl_gen(tl_rd());
                thread_local static std::uniform_real_distribution<double> tl_dist(0.0, 1.0);

                const Individual<GeneType>& parent1 = selection_m.select(current_population);
                const Individual<GeneType>& parent2 = selection_m.select(current_population);

                if (tl_dist(tl_gen) < config_m.crossover_rate) {
                    auto children = crossover_m.crossover(parent1, parent2);
                    new_population[i] = std::move(children.first);
                    if (i + 1 < population_size) {
                        new_population[i + 1] = std::move(children.second);
                    }
                } else {
                    new_population[i] = parent1;
                    if (i + 1 < population_size) {
                        new_population[i + 1] = parent2;
                    }
                }

                mutation_m.mutate(new_population[i], config_m.mutation_rate);
                if (i + 1 < population_size) {
                    mutation_m.mutate(new_population[i + 1], config_m.mutation_rate);
                }
            }
        }

        void initializeCommunication() {
            communicator_m.startReceiving(migration_buffer_m);
        }

        void finalizeCommunication() {
            communicator_m.stopReceiving();
        }

        void handleIncomingMigrants(Population<GeneType>& population) {
            communicator_m.update();

            std::vector<Individual<GeneType>> incoming_migrants;

            migration_buffer_m.popAll(incoming_migrants);
            if (incoming_migrants.empty()) return;

            if (incoming_migrants.size() > max_immigrants_m) {
                std::ranges::nth_element(
                    incoming_migrants,
                    incoming_migrants.begin() + max_immigrants_m,
                    std::less{}
                );

                incoming_migrants.resize(max_immigrants_m);
            }

            deme_replacer_m.replaceDeme(population, std::move(incoming_migrants));
        }

        void handleOutgoingMigrants(const Population<GeneType>& population, const std::size_t generation_idx) {
            if ((generation_idx + 1) % config_m.migration_interval != 0) return;

            auto outgoing_migrants = deme_selector_m.selectDeme(population, config_m.migration_size);

            const std::vector<std::size_t> neighbors = topology_m.getLinks(communicator_m.getRank()).neighbors_out;

            communicator_m.broadcast(outgoing_migrants, neighbors);
        }

        void printGenerationState(const Population<GeneType>& population, const std::size_t generation_idx) const {
            if ((generation_idx + 1) % 50 == 0 || generation_idx == 0) {
                if (communicator_m.getRank() == 0) {
                    std::cout << "Generation " << (generation_idx + 1)
                        << " | Best Fitness: " << population.getBestIndividual().getFitness()
                        << std::endl;
                }
            }
        }

    public:
        IslandGA(
            FitnessFunction<GeneType>& ff,
            Selection<GeneType>& sel,
            Mutation<GeneType>& mu,
            Crossover<GeneType>& cs,
            DemeReplacer<GeneType>& replacer,
            DemeSelector<GeneType>& selector,
            MigrationBuffer<GeneType>& buffer,
            Communicator<GeneType>& comm,
            const Topology& topology,
            const IslandConfig& config,
            const bool elitism = true
        ) : fitness_function_m(ff), selection_m(sel), mutation_m(mu), crossover_m(cs),
            deme_replacer_m(replacer), deme_selector_m(selector), migration_buffer_m(buffer),
            communicator_m(comm), topology_m(topology), config_m(config), use_elitism_m(elitism),
            max_immigrants_m(static_cast<std::size_t>(config.population_size * config.immigration_quota)) {}

        void run(Population<GeneType>& population) {
            if (population.empty()) { return; }

            const std::size_t population_size = population.size();
            const std::size_t num_genes = population.getNumGenes();

            Population<GeneType> new_population(population_size, num_genes);

            evaluatePopulation(population);

            initializeCommunication();

            for (std::size_t generation_idx = 0; generation_idx < config_m.max_generations; ++generation_idx) {
                handleIncomingMigrants(population);

                generateNextGeneration(population, new_population);

                std::swap(population, new_population);

                evaluatePopulation(population);

                handleOutgoingMigrants(population, generation_idx);

                printGenerationState(population, generation_idx);
            }

            finalizeCommunication();
        }
    };
}

#endif // ISLAND_GA_H
