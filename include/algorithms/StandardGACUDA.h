#ifndef STANDARD_GA_CUDA_H
#define STANDARD_GA_CUDA_H

#pragma once

#include "core/Population.h"
#include "core/FitnessFunction.h"
#include "utils/Config.h"

#include <cstddef>
#include <string>

namespace galib {
namespace cuda {

struct StandardGACUDAConfig {
	std::size_t population_size = 100;
	std::size_t dimensions = 2;
	std::size_t max_generations = 1000;

	double mutation_rate = 0.05;
	double crossover_rate = 0.8;
	int tournament_size = 3;

	double lower_bound = -5.12;
	double upper_bound = 5.12;

	bool use_elitism = true;
	std::string problem_name = "Sphere";
	std::string log_file;
};

class StandardGACUDA {
public:
	explicit StandardGACUDA(const StandardGACUDAConfig& config);
	explicit StandardGACUDA(const StandardGACUDAConfig& config, const FitnessFunction<double>& fitness_function);

	bool run(Population<double>& population) const;

	static StandardGACUDAConfig fromConfig(const utils::Config& config);

private:
	StandardGACUDAConfig config_m;
	const FitnessFunction<double>* fitness_function_m = nullptr;
};

bool runStandardGACUDA(const utils::Config& config, Population<double>& population, const FitnessFunction<double>& fitness_function);

}
}

#endif // STANDARD_GA_CUDA_H
