import argparse
import csv
import os
import statistics
import sys
from typing import Iterable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import scripts.benchmark_cuda_cpu as base


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute percentile of empty values")

    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find CUDA crossover points (if any) by sweeping population sizes for each dimension."
        )
    )
    parser.add_argument("--problem", default="HeavyTrig", help="Fitness function name.")
    parser.add_argument(
        "--dims",
        default="64,128,192,256",
        help="Comma-separated dimensions to sweep.",
    )
    parser.add_argument(
        "--pops",
        default="10000,20000,50000,100000,200000",
        help="Comma-separated population sizes to sweep.",
    )
    parser.add_argument("--generations", type=int, default=20, help="Max generations.")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up runs per case.")
    parser.add_argument("--measure", type=int, default=3, help="Measured runs per case.")
    args = parser.parse_args()

    dims = parse_int_list(args.dims)
    pops = parse_int_list(args.pops)

    base.WARMUP_RUNS = args.warmup
    base.MEASURE_RUNS = args.measure
    base.BASE_CONFIG["algorithm"]["max_generations"] = args.generations
    base.BASE_CONFIG["algorithm"]["mutation_rate"] = 0.08
    base.BASE_CONFIG["algorithm"]["crossover_rate"] = 0.9

    base.ensure_dirs()

    print("=== CUDA Crossover Sweep ===")
    print(f"Problem:           {args.problem}")
    print(f"Dimensions:        {dims}")
    print(f"Population sizes:  {pops}")
    print(f"Generations:       {args.generations}")
    print(f"Warmup runs:       {base.WARMUP_RUNS}")
    print(f"Measure runs:      {base.MEASURE_RUNS}")
    print(f"OpenMP executable: {base.EXECUTABLE_OPENMP}")
    print(f"CUDA executable:   {base.EXECUTABLE_CUDA}")

    rows: list[dict[str, object]] = []
    crossovers: dict[int, int | None] = {}

    for dim in dims:
        found_crossover: int | None = None
        print(f"\n--- Dimension {dim} ---")

        for pop in pops:
            cpu_median, _, cpu_runs = base.run_case(args.problem, "OpenMP", dim, pop)
            cuda_median, _, cuda_runs = base.run_case(args.problem, "CUDA", dim, pop)
            speedup = cpu_median / cuda_median if cuda_median > 0 else float("inf")

            cpu_p95 = percentile(cpu_runs, 0.95)
            cuda_p95 = percentile(cuda_runs, 0.95)
            speedup_p95 = cpu_p95 / cuda_p95 if cuda_p95 > 0 else float("inf")

            if found_crossover is None and speedup > 1.0:
                found_crossover = pop

            print(
                f"pop={pop:7d} | CPU median={cpu_median:9.4f}s | CUDA median={cuda_median:9.4f}s "
                f"| speedup={speedup:6.3f} | p95_speedup={speedup_p95:6.3f}"
            )

            rows.append(
                {
                    "problem": args.problem,
                    "dimensions": dim,
                    "pop_size": pop,
                    "cpu_median_seconds": cpu_median,
                    "cuda_median_seconds": cuda_median,
                    "median_speedup_openmp_over_cuda": speedup,
                    "cpu_p95_seconds": cpu_p95,
                    "cuda_p95_seconds": cuda_p95,
                    "p95_speedup_openmp_over_cuda": speedup_p95,
                    "cpu_runs": ";".join(f"{v:.6f}" for v in cpu_runs),
                    "cuda_runs": ";".join(f"{v:.6f}" for v in cuda_runs),
                }
            )

        crossovers[dim] = found_crossover

    csv_path = os.path.join(base.RESULTS_DIR, "cuda_crossover_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "problem",
                "dimensions",
                "pop_size",
                "cpu_median_seconds",
                "cuda_median_seconds",
                "median_speedup_openmp_over_cuda",
                "cpu_p95_seconds",
                "cuda_p95_seconds",
                "p95_speedup_openmp_over_cuda",
                "cpu_runs",
                "cuda_runs",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Crossover Summary ===")
    for dim in dims:
        crossover = crossovers[dim]
        if crossover is None:
            print(f"{dim:4d}D: no crossover in tested range")
        else:
            print(f"{dim:4d}D: CUDA first faster at pop_size={crossover}")

    all_speedups = [float(row["median_speedup_openmp_over_cuda"]) for row in rows]
    print(f"\nMedian of speedups across all points: {statistics.median(all_speedups):.3f}")
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
