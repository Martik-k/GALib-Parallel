#ifndef DEME_SELECTOR_H
#define DEME_SELECTOR_H

#pragma once

#include "core/Individual.h"
#include "core/Population.h"
#include <vector>

namespace galib {

    template<typename GeneType>
    class DemeSelector {
    public:
        virtual ~DemeSelector() = default;

        virtual std::vector<Individual<GeneType>> selectDeme(
            const Population<GeneType>& population,
            std::size_t count
        ) const = 0;
    };

}

#endif // DEME_SELECTOR_H
