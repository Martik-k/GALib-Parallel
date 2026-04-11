#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#pragma once

#include <vector>
#include <limits>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace galib {

    template <typename GeneType = double>
    class Individual {
    private:
        std::vector<GeneType> genotype;
        double fitness;

        static constexpr std::size_t PRINT_LIMIT = 5;
        static constexpr int PRINT_PRECISION = 4;
    public:
        Individual() : fitness(std::numeric_limits<double>::max()) {}

        explicit Individual(std::size_t num_genes)
            : genotype(num_genes), fitness(std::numeric_limits<double>::max()) {}

        [[nodiscard]] std::size_t size() const { return genotype.size(); }

        std::vector<GeneType>& getGenotype() { return genotype; }
        const std::vector<GeneType>& getGenotype() const { return genotype; }
        void setGenotype(const std::vector<GeneType>& new_genotype) { genotype = new_genotype; }
        void setGenotype(std::vector<GeneType>&& new_genotype) noexcept { genotype = std::move(new_genotype); }

        [[nodiscard]] double getFitness() const { return fitness; }
        void setFitness(double new_fitness) { fitness = new_fitness; }

        auto operator<=>(const Individual& other) const {
            return fitness <=> other.fitness;
        }

        bool operator==(const Individual& other) const {
            return fitness == other.fitness;
        }

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
