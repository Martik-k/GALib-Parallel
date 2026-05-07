#ifndef POPULATION_H
#define POPULATION_H

#pragma once

#include "Individual.h"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <omp.h>

namespace galib {

    /**
     * @brief A container for a group of individuals in the genetic algorithm.
     * 
     * Provides methods for initializing individuals with random values and 
     * retrieving the best performing member.
     * 
     * @tparam GeneType The type of the genes.
     */
    template <typename GeneType>
    class Population {
    private:
        std::vector<Individual<GeneType>> individuals;
        std::size_t num_genes_m;
    public:
        /**
         * @brief Constructs a population of a given size.
         * @param size      Number of individuals in the population. Must be > 0.
         * @param num_genes Number of genes each individual should have.
         * @throws std::invalid_argument if size is 0.
         */
        explicit Population(std::size_t size, std::size_t num_genes) : num_genes_m(num_genes) {
            if (size == 0) throw std::invalid_argument("Population size cannot be zero");
            individuals.assign(size, Individual<GeneType>(num_genes));
        }

        /**
         * @brief Randomly initializes all individuals within the specified bounds.
         * 
         * Parallelized using OpenMP.
         * 
         * @param lower Lower bound for each gene.
         * @param upper Upper bound for each gene.
         */
		void initialize(GeneType lower, GeneType upper) {
			#pragma omp parallel
			{
				std::random_device rd;
                std::mt19937 rng(rd());
                std::uniform_real_distribution<GeneType> distribution(lower, upper);

                #pragma omp for schedule(static)
                for (int i = 0; i < static_cast<int>(individuals.size()); ++i) {
                    for (auto& gene : individuals[i].getGenotype()) {
                        gene = distribution(rng);
                    }
                }
			}
		}

        /**
         * @brief Returns the total number of individuals in the population.
         * @return Population size.
         */
        [[nodiscard]] std::size_t size() const { return individuals.size(); }

        /**
         * @brief Checks if the population is empty.
         * @return True if size is 0, false otherwise.
         */
        [[nodiscard]] bool empty() const { return individuals.empty(); }

        /**
         * @brief Returns the number of genes for individuals in this population.
         * @return Gene count.
         */
        [[nodiscard]] std::size_t getNumGenes() const { return num_genes_m; }

        /**
         * @brief Accesses an individual by its index.
         * @param index Zero-based index.
         * @return Reference to the individual.
         */
        Individual<GeneType>& operator [](std::size_t index) { return individuals[index]; }

        /**
         * @brief Accesses an individual by its index (const version).
         * @param index Zero-based index.
         * @return Constant reference to the individual.
         */
        const Individual<GeneType>& operator [](std::size_t index) const { return individuals[index]; }

        /** @brief Returns an iterator to the beginning. */
        typename std::vector<Individual<GeneType>>::iterator begin() { return individuals.begin(); }

        /** @brief Returns an iterator to the end. */
        typename std::vector<Individual<GeneType>>::iterator end() { return individuals.end(); }

        /** @brief Returns a constant iterator to the beginning. */
        typename std::vector<Individual<GeneType>>::const_iterator begin() const { return individuals.begin(); }

        /** @brief Returns a constant iterator to the end. */
        typename std::vector<Individual<GeneType>>::const_iterator end() const { return individuals.end(); }

        /**
         * @brief Finds and returns the individual with the best fitness score.
         * @return Constant reference to the best individual.
         */
        const Individual<GeneType>& getBestIndividual() const {
            return *std::min_element(individuals.begin(), individuals.end());
        }
    };
}

#endif // POPULATION_H
