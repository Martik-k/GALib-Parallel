#ifndef FITNESS_FACTORY_H
#define FITNESS_FACTORY_H

#pragma once

#include <memory>
#include <string>
#include <stdexcept>

#include "core/FitnessFunction.h"
#include "benchmarks/SphereFunction.h"
#include "benchmarks/RastriginFunction.h"

#include "utils/Config.h"

namespace galib {
    namespace utils {
        class FitnessFactory {
        public:
            static std::unique_ptr<FitnessFunction<double>> create(const Config::Problem& prob_config) {
                if (prob_config.name == "Sphere") {
                    return std::make_unique<benchmark::SphereFunction<double>>(
                        prob_config.dimensions, prob_config.lower_bound, prob_config.upper_bound);
                } else if (prob_config.name == "Rastrigin") {
                    return std::make_unique<benchmark::RastriginFunction<double>>(
                        prob_config.dimensions, prob_config.lower_bound, prob_config.upper_bound);
                }
                throw std::invalid_argument("Unknown function name in config: " + prob_config.name);
            }
        };
    }
}

#endif // FITNESS_FACTORY_H