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
#include "algorithms/island/communication/buffers/MigrationBuffer.h"
#include "algorithms/island/communication/communicators/Communicator.h"
#include "algorithms/island/communication/serializers/Serializer.h"
#include "algorithms/island/IslandConfig.h"
#include "algorithms/island/topology/Topology.h"
#include "utils/StateLogger.h"

#include <iostream>
#include <random>
#include <utility>
#include <algorithm>


namespace galib {
    template <typename GeneType = double>
    class IslandGA {
    private:
        FitnessFunction<GeneType>& fitness_function_m;
        
        std::unique_ptr<Selection<GeneType>> selection_m;
        std::unique_ptr<Mutation<GeneType>> mutation_m;
        std::unique_ptr<Crossover<GeneType>> crossover_m;
        std::unique_ptr<DemeReplacer<GeneType>> deme_replacer_m;
        std::unique_ptr<DemeSelector<GeneType>> deme_selector_m;
        
        std::unique_ptr<MigrationBuffer<GeneType>> migration_buffer_m;
        std::unique_ptr<Communicator<GeneType>> communicator_m;
        std::unique_ptr<const Topology> topology_m;
        std::unique_ptr<Serializer<GeneType>> serializer_m;

        const IslandConfig config_m;

        std::unique_ptr<utils::StateLogger<GeneType>> logger_m;
        std::size_t log_interval_m = 0;
        bool verbose_m = false;
        std::size_t console_log_interval_m = 0;

        bool use_elitism_m;

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
            std::size_t elitism_offset = 0;

            if (use_elitism_m) {
                new_population[0] = current_population.getBestIndividual();
                elitism_offset = 1;
            }

            #pragma omp parallel for schedule(static)
            for (std::size_t i = elitism_offset; i < population_size; i += 2) {
                thread_local static std::random_device tl_rd;
                thread_local static std::mt19937_64 tl_gen(tl_rd());
                thread_local static std::uniform_real_distribution<double> tl_dist(0.0, 1.0);

                const Individual<GeneType>& parent1 = selection_m->select(current_population);
                const Individual<GeneType>& parent2 = selection_m->select(current_population);

                if (tl_dist(tl_gen) < config_m.crossover_rate) {
                    auto children = crossover_m->crossover(parent1, parent2);
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

                mutation_m->mutate(new_population[i], config_m.mutation_rate);
                if (i + 1 < population_size) {
                    mutation_m->mutate(new_population[i + 1], config_m.mutation_rate);
                }
            }
        }

        void initializeCommunication() {
            communicator_m->startReceiving(*migration_buffer_m);
        }

        void finalizeCommunication() {
            communicator_m->stopReceiving();
        }

        void handleIncomingMigrants(Population<GeneType>& population) {
            communicator_m->update();

            std::vector<Individual<GeneType>> incoming_migrants;

            migration_buffer_m->popAll(incoming_migrants);
            if (incoming_migrants.empty()) return;

            const std::size_t max_immigrants = static_cast<std::size_t>(
                static_cast<double>(population.size()) * config_m.immigration_quota
            );

            if (incoming_migrants.size() > max_immigrants) {
                std::ranges::nth_element(
                    incoming_migrants,
                    incoming_migrants.begin() + max_immigrants,
                    std::less{}
                );

                incoming_migrants.resize(max_immigrants);
            }

            deme_replacer_m->replaceDeme(population, std::move(incoming_migrants));
        }

        void handleOutgoingMigrants(const Population<GeneType>& population, const std::size_t generation_idx) {
            if ((generation_idx + 1) % config_m.migration_interval != 0) return;

            auto outgoing_migrants = deme_selector_m->selectDeme(population, config_m.migration_size);

            const std::vector<std::size_t> neighbors = topology_m->getLinks(communicator_m->getRank()).neighbors_out;


            communicator_m->broadcast(outgoing_migrants, neighbors);
        }

        void synchronizeGlobalBest(Population<GeneType>& population) {
            const Individual<GeneType> global_best = communicator_m->allReduceBest(population.getBestIndividual());

            if (global_best.getFitness() < population.getBestIndividual().getFitness()) {
                population[0] = std::move(global_best);
            }
        }

        void logState(const Population<GeneType>& population, const std::size_t generation_idx) const {
            if (verbose_m && communicator_m->getRank() == 0) {
                std::cout << "Generation " << (generation_idx + 1)
                          << " | Global Best Fitness: " << population.getBestIndividual().getFitness()
                          << std::endl;
            }
        }

    public:
        IslandGA(
            FitnessFunction<GeneType>& ff,
            std::unique_ptr<Selection<GeneType>> sel,
            std::unique_ptr<Mutation<GeneType>> mu,
            std::unique_ptr<Crossover<GeneType>> cs,
            std::unique_ptr<DemeReplacer<GeneType>> replacer,
            std::unique_ptr<DemeSelector<GeneType>> selector,
            std::unique_ptr<MigrationBuffer<GeneType>> buffer,
            std::unique_ptr<Communicator<GeneType>> comm,
            std::unique_ptr<const Topology> topology,
            std::unique_ptr<Serializer<GeneType>> serializer,
            const IslandConfig& config,
            const bool elitism = true
        ) : fitness_function_m(ff), 
            selection_m(std::move(sel)), 
            mutation_m(std::move(mu)), 
            crossover_m(std::move(cs)),
            deme_replacer_m(std::move(replacer)), 
            deme_selector_m(std::move(selector)), 
            migration_buffer_m(std::move(buffer)),
            communicator_m(std::move(comm)), 
            topology_m(std::move(topology)), 
            serializer_m(std::move(serializer)),
            config_m(config), 
            use_elitism_m(elitism) {}

        void enableConsoleOutput(const bool enabled, const std::size_t interval) {
            verbose_m = enabled;
            console_log_interval_m = interval;
        }

        void enableFileLogging(const std::string& directory, const std::size_t interval) {
            logger_m = std::make_unique<utils::StateLogger<GeneType>>(directory, communicator_m->getRank());
            log_interval_m = interval;
        }

        void run(Population<GeneType>& population) {
            if (population.empty()) { return; }

            const std::size_t population_size = population.size();
            const std::size_t num_genes = population.getNumGenes();

            Population<GeneType> new_population(population_size, num_genes);

            evaluatePopulation(population);

            initializeCommunication();

            if (logger_m && log_interval_m > 0) {
                logger_m->writeHeader(num_genes);
                logger_m->log(population, 0);
            }

            for (std::size_t generation_idx = 0; generation_idx < config_m.max_generations; ++generation_idx) {
                handleIncomingMigrants(population);

                generateNextGeneration(population, new_population);

                std::swap(population, new_population);

                evaluatePopulation(population);

                handleOutgoingMigrants(population, generation_idx);

                if (logger_m && log_interval_m > 0 && (generation_idx + 1) % log_interval_m == 0) {
                    logger_m->log(population, generation_idx + 1);
                }

                if ((generation_idx + 1) % console_log_interval_m == 0 || generation_idx == 0) {
                    logState(population, generation_idx);
                }
            }

            finalizeCommunication();

            synchronizeGlobalBest(population);
        }
    };
}

#endif // ISLAND_GA_H
