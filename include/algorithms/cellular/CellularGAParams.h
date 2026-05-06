#ifndef CELLULAR_GA_PARAMS_H
#define CELLULAR_GA_PARAMS_H

#pragma once

#include <cstddef>

#include "algorithms/AlgorithmConfig.h"

namespace galib {

    /**
     * @brief Parameters for the Cellular Genetic Algorithm.
     */
    struct CellularGAParams : public AlgorithmConfig {
        bool use_local_elitism = true;
    };

} // namespace galib

#endif // CELLULAR_GA_PARAMS_H
