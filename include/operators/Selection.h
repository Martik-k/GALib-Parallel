#ifndef SELECTION_H
#define SELECTION_H

#pragma once

#include "../core/Individual.h"
#include "../core/Population.h"

namespace galib {

    template <typename GeneType>
    class Selection {
    public:
        virtual ~Selection() = default;
        virtual const Individual<GeneType>& select(const Population<GeneType>& population) = 0;
    };

}

#endif // SELECTION_H