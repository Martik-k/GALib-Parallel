#ifndef MUTATION_H
#define MUTATION_H

#pragma once

#include "../core/Individual.h"

namespace galib {

    template <typename GeneType>
    class Mutation {
    public:
        virtual ~Mutation() = default;
        virtual void mutate(Individual<GeneType>& individual, double mutation_rate) = 0;
    };

}

#endif // MUTATION_H