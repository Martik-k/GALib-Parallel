#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#pragma once

#include <vector>
#include <limits>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace galib {

    /**
     * @brief Represents a single candidate solution in the population.
     * 
     * An Individual consists of a genotype (a vector of genes) and a fitness score.
     * It supports comparison operators for sorting based on fitness (minimization).
     * 
     * @tparam GeneType The type of the genes (default: double).
     */
    template <typename GeneType = double>
    class Individual {
    private:
        std::vector<GeneType> genotype;
        double fitness;

        static constexpr std::size_t PRINT_LIMIT = 5;
        static constexpr int PRINT_PRECISION = 4;
    public:
        /**
         * @brief Default constructor. Initializes fitness to the maximum possible double value.
         */
        Individual() : fitness(std::numeric_limits<double>::max()) {}

        /**
         * @brief Constructs an Individual with a specific number of genes.
         * @param num_genes Number of genes in the genotype.
         */
        explicit Individual(std::size_t num_genes)
            : genotype(num_genes), fitness(std::numeric_limits<double>::max()) {}

        /**
         * @brief Returns the number of genes in the genotype.
         * @return Size of the genotype vector.
         */
        [[nodiscard]] std::size_t size() const { return genotype.size(); }

        /**
         * @brief Returns a reference to the genotype vector.
         * @return Reference to the genotype.
         */
        std::vector<GeneType>& getGenotype() { return genotype; }

        /**
         * @brief Returns a constant reference to the genotype vector.
         * @return Constant reference to the genotype.
         */
        const std::vector<GeneType>& getGenotype() const { return genotype; }

        /**
         * @brief Sets the genotype by copying.
         * @param new_genotype The new genotype vector.
         */
        void setGenotype(const std::vector<GeneType>& new_genotype) { genotype = new_genotype; }

        /**
         * @brief Sets the genotype by moving.
         * @param new_genotype The new genotype vector.
         */
        void setGenotype(std::vector<GeneType>&& new_genotype) noexcept { genotype = std::move(new_genotype); }

        /**
         * @brief Gets the current fitness score.
         * @return Fitness value.
         */
        [[nodiscard]] double getFitness() const { return fitness; }

        /**
         * @brief Sets a new fitness score.
         * @param new_fitness The new fitness value.
         */
        void setFitness(double new_fitness) { fitness = new_fitness; }

        /**
         * @brief Three-way comparison operator based on fitness (for minimization).
         * @param other The individual to compare against.
         */
        auto operator<=>(const Individual& other) const {
            return fitness <=> other.fitness;
        }

        /**
         * @brief Equality operator based on fitness.
         * @param other The individual to compare against.
         */
        bool operator==(const Individual& other) const {
            return fitness == other.fitness;
        }

        /**
         * @brief Streams a summary of the individual (fitness and first few genes).
         */
        friend std::ostream& operator<<(std::ostream& os, const Individual& ind) {
            os << "[   ";

            std::size_t limit = std::min(ind.genotype.size(), PRINT_LIMIT);
            for (std::size_t i = 0; i < limit; ++i) {
                os << std::fixed << std::setprecision(PRINT_PRECISION) << ind.genotype[i] << "   ";
            }
            if (ind.genotype.size() > PRINT_LIMIT) os << "... ";
            os << "]";

            return os;
        }
    };

}

#endif // INDIVIDUAL_H
