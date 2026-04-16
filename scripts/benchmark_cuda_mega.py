import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import scripts.benchmark_cuda_cpu as base


def main() -> None:
    # Mega-GA scenario aimed at CUDA viability:
    # - expensive fitness (HeavyTrig)
    # - massive population (100000)
    # - stay-on-device (no per-generation logging)
    base.WARMUP_RUNS = 1
    base.MEASURE_RUNS = 3

    base.BASE_CONFIG["algorithm"]["max_generations"] = 10
    base.BASE_CONFIG["algorithm"]["mutation_rate"] = 0.08
    base.BASE_CONFIG["algorithm"]["crossover_rate"] = 0.9

    problem = "HeavyTrig"
    dimensions = 128
    pop_size = 100000

    print("Running Mega-GA CUDA scenario...")
    print(f"Problem={problem}, dimensions={dimensions}, pop_size={pop_size}, generations=10")

    cpu_t, _, cpu_runs = base.run_case(problem, "OpenMP", dimensions, pop_size)
    cuda_t, _, cuda_runs = base.run_case(problem, "CUDA", dimensions, pop_size)
    speedup = cpu_t / cuda_t if cuda_t > 0 else float("inf")

    print(f"OpenMP median: {cpu_t:.4f}s | runs={[round(v, 4) for v in cpu_runs]}")
    print(f"CUDA   median: {cuda_t:.4f}s | runs={[round(v, 4) for v in cuda_runs]}")
    print(f"Speedup (OpenMP/CUDA): {speedup:.3f}")


if __name__ == "__main__":
    main()
