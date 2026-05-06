#ifndef STANDARD_GA_PARAMS_H
#define STANDARD_GA_PARAMS_H

#pragma once

#include <cstddef>

#include "algorithms/AlgorithmConfig.h"

namespace galib {

    /**
     * @brief Parameters for the Standard (Generational) Genetic Algorithm.
     */
    struct StandardGAParams : public AlgorithmConfig {
        // All fields inherited from AlgorithmConfig
    };

} // namespace galib

#endif // STANDARD_GA_PARAMS_H
