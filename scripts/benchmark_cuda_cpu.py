import csv
import os
import subprocess
import tempfile
import time

import matplotlib.pyplot as plt
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
EXECUTABLE = os.path.join(BUILD_DIR, "ga_example")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "visualizations", "images")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "visualizations", "results")


FUNCTIONS = ["Sphere", "Rastrigin"]
SIZES = [
    {"label": "Small", "dimensions": 2, "pop_size": 250},
    {"label": "Medium", "dimensions": 10, "pop_size": 2000},
    {"label": "Large", "dimensions": 50, "pop_size": 10000},
]

BASE_CONFIG = {
    "algorithm": {
        "max_generations": 50,
        "mutation_rate": 0.1,
        "crossover_rate": 0.9,
        "selection": {"type": "Tournament", "tournament_size": 3},
    },
    "output": {"log_file": ""},
}


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def run_case(problem_name: str, backend: str, dimensions: int, pop_size: int) -> tuple[float, str]:
    config = {
        "problem": {
            "name": problem_name,
            "dimensions": dimensions,
            "lower_bound": -5.12,
            "upper_bound": 5.12,
        },
        "algorithm": {
            **BASE_CONFIG["algorithm"],
            "backend": backend,
            "pop_size": pop_size,
        },
        "output": {"log_file": ""},
    }

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        config_path = handle.name

    env = os.environ.copy()
    if backend.upper() == "OPENMP":
        env.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))

    start = time.perf_counter()
    process = subprocess.run(
        [EXECUTABLE, config_path],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start

    os.unlink(config_path)

    if process.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed for {problem_name} / {backend} / {dimensions}D / {pop_size}:\n"
            f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        )

    return elapsed, process.stdout


def main() -> None:
    if not os.path.exists(EXECUTABLE):
        raise FileNotFoundError(f"Executable not found: {EXECUTABLE}")

    ensure_dirs()

    results = []
    print("Starting CPU vs CUDA benchmark matrix...")

    for problem_name in FUNCTIONS:
        for size in SIZES:
            for backend in ["OpenMP", "CUDA"]:
                elapsed, stdout = run_case(
                    problem_name,
                    backend,
                    size["dimensions"],
                    size["pop_size"],
                )
                best_line = next((line for line in stdout.splitlines() if "Best Fitness Found" in line), "")
                print(f"{problem_name:9s} | {size['label']:6s} | {backend:6s} | {elapsed:8.4f}s | {best_line}")
                results.append(
                    {
                        "function": problem_name,
                        "size": size["label"],
                        "dimensions": size["dimensions"],
                        "pop_size": size["pop_size"],
                        "backend": backend,
                        "time_seconds": elapsed,
                        "best_line": best_line,
                    }
                )

    csv_path = os.path.join(RESULTS_DIR, "cuda_cpu_benchmark_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["function", "size", "dimensions", "pop_size", "backend", "time_seconds", "best_line"],
        )
        writer.writeheader()
        writer.writerows(results)

    plots = []
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for axis, problem_name in zip(axes, FUNCTIONS):
        subset = [row for row in results if row["function"] == problem_name]
        x_labels = [size["label"] for size in SIZES]
        openmp_times = [next(row["time_seconds"] for row in subset if row["size"] == label and row["backend"] == "OpenMP") for label in x_labels]
        cuda_times = [next(row["time_seconds"] for row in subset if row["size"] == label and row["backend"] == "CUDA") for label in x_labels]

        positions = range(len(x_labels))
        width = 0.35
        axis.bar([p - width / 2 for p in positions], openmp_times, width=width, label="CPU / OpenMP", color="#3498db")
        axis.bar([p + width / 2 for p in positions], cuda_times, width=width, label="CUDA", color="#2ecc71")
        axis.set_title(f"{problem_name}")
        axis.set_xticks(list(positions))
        axis.set_xticklabels(x_labels)
        axis.set_ylabel("Seconds")
        axis.grid(True, linestyle="--", alpha=0.3, axis="y")
        axis.legend()

        plots.append(
            {
                "function": problem_name,
                "openmp": openmp_times,
                "cuda": cuda_times,
            }
        )

    plt.suptitle("CPU vs CUDA RCGA Runtime Comparison")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "cuda_cpu_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")
    print(f"Saved table to {csv_path}")


if __name__ == "__main__":
    main()