import os
import sys
import statistics

import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "visualizations", "images")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "visualizations", "results")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import scripts.benchmark_cuda_cpu as base


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def main() -> None:
    ensure_dirs()

    # Heavy, compute-dominated stress test. The goal is to make GPU arithmetic
    # matter more than orchestration overhead.
    base.WARMUP_RUNS = 1
    base.MEASURE_RUNS = 3
    base.BASE_CONFIG["algorithm"]["max_generations"] = 5
    base.BASE_CONFIG["algorithm"]["mutation_rate"] = 0.08
    base.BASE_CONFIG["algorithm"]["crossover_rate"] = 0.9

    problem_name = "HeavyTrig"
    sizes = [
        {"label": "Stress", "dimensions": 64, "pop_size": 10000},
    ]

    results = []
    print("Starting heavy CUDA-favoring benchmark matrix...")

    for size in sizes:
        cpu_time, cpu_stdout, cpu_runs = base.run_case(
            problem_name,
            "OpenMP",
            size["dimensions"],
            size["pop_size"],
        )
        cuda_time, cuda_stdout, cuda_runs = base.run_case(
            problem_name,
            "CUDA",
            size["dimensions"],
            size["pop_size"],
        )

        cpu_best = next((line for line in cpu_stdout.splitlines() if "Best Fitness Found" in line), "")
        cuda_best = next((line for line in cuda_stdout.splitlines() if "Best Fitness Found" in line), "")
        speedup = cpu_time / cuda_time if cuda_time > 0 else float("inf")

        print(
            f"{problem_name:9s} | {size['label']:6s} | CPU(1t)={cpu_time:8.4f}s {cpu_runs} | "
            f"CUDA={cuda_time:8.4f}s {cuda_runs} | speedup(OMP/CUDA)={speedup:.3f}"
        )

        results.append({
            "size": size["label"],
            "cpu": cpu_time,
            "cuda": cuda_time,
            "speedup": speedup,
            "cpu_best": cpu_best,
            "cuda_best": cuda_best,
        })

    fig, axis = plt.subplots(figsize=(8, 5))
    positions = range(len(results))
    cpu_times = [row["cpu"] for row in results]
    cuda_times = [row["cuda"] for row in results]
    width = 0.35

    axis.bar([p - width / 2 for p in positions], cpu_times, width=width, label="CPU / OpenMP (1 thread)", color="#3498db")
    axis.bar([p + width / 2 for p in positions], cuda_times, width=width, label="CUDA", color="#2ecc71")
    axis.set_xticks(list(positions))
    axis.set_xticklabels([row["size"] for row in results])
    axis.set_ylabel("Seconds")
    axis.set_title("HeavyTrig stress benchmark")
    axis.grid(True, linestyle="--", alpha=0.3, axis="y")
    axis.legend()

    plot_path = os.path.join(OUTPUT_DIR, "cuda_heavy_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)

    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()