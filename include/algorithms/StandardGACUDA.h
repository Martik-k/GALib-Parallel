#ifndef STANDARD_GA_CUDA_H
#define STANDARD_GA_CUDA_H

#pragma once

#include "core/Population.h"
#include "utils/Config.h"

namespace galib {
namespace cuda {

bool runStandardGACUDA(const utils::Config& config, Population<double>& population);

}
}

#endif // STANDARD_GA_CUDA_H
