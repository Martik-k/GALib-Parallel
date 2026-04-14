#ifndef DE_PARAMS_H
#define DE_PARAMS_H

#pragma once

#include <cstddef>

namespace galib {

    /**
     * @brief Parameters for the Differential Evolution Genetic Algorithm.
     */
    struct DEParams {
        double f_weight = 0.8;
        double cr_rate = 0.9;         // Maps to crossover_rate in YAML
        std::size_t max_generations = 100;
    };

} // namespace galib

#endif // DE_PARAMS_H
