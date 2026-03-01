#ifndef CROSSOVER_H
#define CROSSOVER_H

#pragma once

#include "../core/Individual.h"
#include <utility>

namespace galib {

    template <typename GeneType>
    class Crossover {
    public:
        virtual ~Crossover() = default;
        virtual std::pair<Individual<GeneType>, Individual<GeneType>> crossover (
            const Individual<GeneType>& parent1, const Individual<GeneType>& parent2) = 0;
    };

}

#endif // CROSSOVER_H