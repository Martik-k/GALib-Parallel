#ifndef DE_PARAMS_H
#define DE_PARAMS_H

#pragma once

#include <cstddef>

#include "algorithms/AlgorithmConfig.h"

namespace galib {

    /**
     * @brief Parameters for the Differential Evolution Genetic Algorithm.
     */
    struct DEParams : public AlgorithmConfig {
        double f_weight = 0.8;
    };

} // namespace galib

#endif // DE_PARAMS_H
