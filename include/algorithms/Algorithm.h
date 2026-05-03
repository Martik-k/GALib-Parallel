#ifndef ALGORITHM_H
#define ALGORITHM_H

#pragma once

#include "core/Population.h"

namespace galib {

    /**
     * @brief Abstract base class for all optimization algorithms.
     * @tparam GeneType The type of the genes (e.g., double, int).
     */
    template <typename GeneType>
    class Algorithm {
    public:
        virtual ~Algorithm() = default;

        /**
         * @brief Executes the optimization process on the given population.
         * @param population The population to optimize.
         */
        virtual void run(Population<GeneType>& population) = 0;
    };

} // namespace galib

#endif // ALGORITHM_H
