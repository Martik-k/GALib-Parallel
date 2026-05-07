#ifndef CUDA_EVALUATOR_H
#define CUDA_EVALUATOR_H

#pragma once

#include <functional>
#include <string>
#include <vector>

namespace galib::cuda::internal {

/**
 * @brief Parameters for a full GPU-accelerated evolution run.
 *
 * The GPU handles selection, crossover, and mutation entirely on device.
 * Fitness evaluation runs on GPU for known problem names; otherwise the
 * @p fitness_callback is invoked on the host (CPU) each generation.
 */
struct CUDARunParams {
    int    pop_size;
    int    dims;
    int    max_generations;
    float  mutation_rate;
    float  crossover_rate;
    bool   use_elitism;
    int    tournament_size = 3;
    float  sigma          = 0.1f; ///< Gaussian mutation standard deviation.

    /** Short name returned by FitnessFunction::name(). Empty = use callback. */
    std::string problem_name;

    /**
     * CPU fitness callback used when problem_name is unknown.
     * Receives a flat float array [dims] for one individual, returns fitness.
     */
    std::function<double(const float*, int)> fitness_callback;

    /** Called after each generation with (generation_index, best_fitness). */
    std::function<void(int, double)> progress_callback;
};

/**
 * @brief Runs the full GA evolution loop on the GPU.
 *
 * @param params     Configuration and callbacks (see CUDARunParams).
 * @param h_genes    In: initial population as flat row-major float array [pop_size * dims].
 *                   Out: final population after evolution.
 * @param h_fitness  Out: fitness values for the final population [pop_size].
 * @return true on success, false if a CUDA error occurred (caller should fall back to CPU).
 */
bool runCUDAEvolution(CUDARunParams& params,
                      std::vector<float>& h_genes,
                      std::vector<float>& h_fitness);

} // namespace galib::cuda::internal

#endif // CUDA_EVALUATOR_H
