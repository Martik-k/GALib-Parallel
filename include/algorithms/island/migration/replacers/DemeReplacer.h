#ifndef DEME_REPLACER_H
#define DEME_REPLACER_H

#pragma once

#include "core/Individual.h"
#include "core/Population.h"
#include <vector>

namespace galib {

    template<typename GeneType>
    class DemeReplacer {
    public:
        virtual ~DemeReplacer() = default;

        virtual void replaceDeme(
            Population<GeneType>& population,
            std::vector<Individual<GeneType>>&& deme
        ) const = 0;
    };

}

#endif // DEME_REPLACER_H
