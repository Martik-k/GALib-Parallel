"""
GALib-Parallel CUDA Benchmark
------------------------------
Runs build/benchmark_ga_cuda, collects results, and generates analysis plots.

Usage (from project root):
    python3 scripts/CUDA_benchmark/benchmark.py

Outputs:
    scripts/CUDA_benchmark/results.csv   — raw numbers
    scripts/CUDA_benchmark/plots/        — all PNG figures
"""

import subprocess
import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent.parent
BIN      = ROOT / "build" / "benchmark_ga_cuda"
CSV_PATH = Path(__file__).parent / "results.csv"
PLOT_DIR = Path(__file__).parent / "plots"

COLORS = {
    "gpu_kernel":   "#2196F3",   # blue  — full GPU
    "cpu_callback": "#FF9800",   # orange — hybrid
    "cpu_only":     "#9E9E9E",   # grey  — CPU baseline
}
LABEL = {
    "gpu_kernel":   "GPU kernel (full GPU)",
    "cpu_callback": "CPU callback (hybrid)",
    "cpu_only":     "CPU only",
}

# ── run benchmark ─────────────────────────────────────────────────────────────
def run_benchmark():
    if not BIN.exists():
        sys.exit(f"[ERROR] Binary not found: {BIN}\n"
                 "  Build first: cmake --build build --target benchmark_ga_cuda")

    print(f"Running {BIN.name} …")
    result = subprocess.run(
        [str(BIN), "--csv", str(CSV_PATH)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.exit(f"[ERROR] Benchmark failed:\n{result.stderr}")
    print(result.stdout)

# ── load CSV ──────────────────────────────────────────────────────────────────
def load_results():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "function":    r["function"],
                "eval_type":   r["eval_type"],
                "pop_size":    int(r["pop_size"]),
                "dims":        int(r["dims"]),
                "cpu_ms":      float(r["cpu_ms"]),
                "cpu_fitness": float(r["cpu_fitness"]),
                "gpu_ms":      float(r["gpu_ms"]),
                "gpu_fitness": float(r["gpu_fitness"]),
                "speedup":     float(r["speedup"]),
            })
    return rows

# ── helpers ───────────────────────────────────────────────────────────────────
def filter_rows(rows, **kwargs):
    out = rows
    for k, v in kwargs.items():
        out = [r for r in out if r[k] == v]
    return out

def save(fig, name):
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.relative_to(ROOT)}")

# ── plots ─────────────────────────────────────────────────────────────────────

def plot_speedup_vs_popsize(rows):
    """Line chart: speedup vs population size for each function × dims combination."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("GPU Speedup vs Population Size", fontsize=14, fontweight="bold")

    for ax, dims in zip(axes, [10, 50]):
        ax.set_title(f"Dimensions = {dims}")
        ax.set_xlabel("Population size")
        ax.set_ylabel("Speedup (CPU time / GPU time)")
        ax.axhline(1, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.3)

        for func in ["Rastrigin", "Sphere", "Ackley"]:
            subset = filter_rows(rows, function=func, dims=dims)
            if not subset:
                continue
            xs = [r["pop_size"] for r in subset]
            ys = [r["speedup"]  for r in subset]
            et = subset[0]["eval_type"]
            ax.plot(xs, ys, marker="o", label=f"{func} ({LABEL[et]})",
                    color=COLORS[et], linewidth=2,
                    linestyle="-" if func != "Ackley" else "--")

        ax.legend(fontsize=8)
        ax.set_xscale("log")

    fig.tight_layout()
    save(fig, "speedup_vs_popsize.png")


def plot_absolute_times(rows):
    """Grouped bar: CPU time vs GPU time for every (function, pop_size, dims) config."""
    # Use dims=50 slice for a representative view
    subset = filter_rows(rows, dims=50)
    funcs   = sorted({r["function"] for r in subset})
    pops    = sorted({r["pop_size"] for r in subset})

    fig, axes = plt.subplots(1, len(funcs), figsize=(5 * len(funcs), 5), sharey=False)
    if len(funcs) == 1:
        axes = [axes]
    fig.suptitle("Wall Time: CPU vs GPU (dims=50)", fontsize=14, fontweight="bold")

    for ax, func in zip(axes, funcs):
        data = filter_rows(subset, function=func)
        data.sort(key=lambda r: r["pop_size"])
        xs     = np.arange(len(data))
        width  = 0.35
        cpu_ts = [r["cpu_ms"] / 1000 for r in data]
        gpu_ts = [r["gpu_ms"] / 1000 for r in data]
        labels = [str(r["pop_size"]) for r in data]
        et     = data[0]["eval_type"]

        ax.bar(xs - width/2, cpu_ts, width, label="CPU",
               color="#9E9E9E", edgecolor="white")
        ax.bar(xs + width/2, gpu_ts, width, label=f"GPU ({LABEL[et]})",
               color=COLORS[et], edgecolor="white")
        ax.set_title(func)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_xlabel("Population size")
        ax.set_ylabel("Wall time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "absolute_times.png")


def plot_speedup_heatmap(rows):
    """Heatmap: speedup for each (pop_size × dims) grid per function."""
    funcs = sorted({r["function"] for r in rows})
    fig, axes = plt.subplots(1, len(funcs), figsize=(5 * len(funcs), 4))
    if len(funcs) == 1:
        axes = [axes]
    fig.suptitle("Speedup Heatmap (pop size × dims)", fontsize=14, fontweight="bold")

    for ax, func in zip(axes, funcs):
        data  = filter_rows(rows, function=func)
        pops  = sorted({r["pop_size"] for r in data})
        dims  = sorted({r["dims"]     for r in data})
        grid  = np.zeros((len(dims), len(pops)))
        for r in data:
            i = dims.index(r["dims"])
            j = pops.index(r["pop_size"])
            grid[i, j] = r["speedup"]

        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", origin="lower")
        ax.set_xticks(range(len(pops)));  ax.set_xticklabels(pops, rotation=30)
        ax.set_yticks(range(len(dims)));  ax.set_yticklabels(dims)
        ax.set_xlabel("Population size");  ax.set_ylabel("Dimensions")
        ax.set_title(func)
        for i in range(len(dims)):
            for j in range(len(pops)):
                ax.text(j, i, f"{grid[i,j]:.0f}x",
                        ha="center", va="center", fontsize=8,
                        color="black" if grid[i,j] < grid.max() * 0.7 else "white")
        fig.colorbar(im, ax=ax, label="Speedup")

    fig.tight_layout()
    save(fig, "speedup_heatmap.png")


def plot_eval_type_comparison(rows):
    """Bar chart comparing max speedup for GPU-kernel vs CPU-callback functions."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Max Speedup by Evaluation Strategy", fontsize=14, fontweight="bold")

    funcs_by_type = {}
    for r in rows:
        et = r["eval_type"]
        fn = r["function"]
        if et not in funcs_by_type:
            funcs_by_type[et] = {}
        funcs_by_type[et][fn] = max(funcs_by_type[et].get(fn, 0), r["speedup"])

    all_funcs = sorted({r["function"] for r in rows})
    xs        = np.arange(len(all_funcs))
    width     = 0.6

    for func, x in zip(all_funcs, xs):
        for et, func_map in funcs_by_type.items():
            if func in func_map:
                sp = func_map[func]
                color = COLORS.get(et, "#607D8B")
                bar = ax.bar(x, sp, width, color=color, edgecolor="white",
                             label=LABEL.get(et, et))
                ax.text(x, sp + 3, f"{sp:.0f}x",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(xs);  ax.set_xticklabels(all_funcs)
    ax.set_ylabel("Max speedup (CPU time / GPU time)")
    ax.set_xlabel("Fitness function")
    ax.grid(True, axis="y", alpha=0.3)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=9)

    fig.tight_layout()
    save(fig, "eval_type_comparison.png")


def plot_fitness_quality(rows):
    """Scatter: CPU best fitness vs GPU best fitness — how similar is quality?"""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Solution Quality: CPU vs GPU", fontsize=14, fontweight="bold")

    for r in rows:
        if r["eval_type"] == "cpu_only":
            continue
        ax.scatter(r["cpu_fitness"], r["gpu_fitness"],
                   color=COLORS[r["eval_type"]], alpha=0.7, s=60,
                   label=f"{r['function']} ({LABEL[r['eval_type']]})")

    # Perfect agreement diagonal
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5, label="CPU = GPU")
    ax.set_xlabel("CPU best fitness")
    ax.set_ylabel("GPU best fitness")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=8)

    fig.tight_layout()
    save(fig, "fitness_quality.png")


def plot_scaling(rows):
    """Log-log: speedup vs total genes (pop_size * dims) per function."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Speedup vs Problem Scale (pop × dims)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total genes (pop_size × dims)  [log scale]")
    ax.set_ylabel("Speedup  [log scale]")
    ax.axhline(1, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xscale("log");  ax.set_yscale("log")

    for func in sorted({r["function"] for r in rows}):
        subset = filter_rows(rows, function=func)
        if subset[0]["eval_type"] == "cpu_only":
            continue
        subset.sort(key=lambda r: r["pop_size"] * r["dims"])
        xs = [r["pop_size"] * r["dims"] for r in subset]
        ys = [r["speedup"]             for r in subset]
        et = subset[0]["eval_type"]
        ax.plot(xs, ys, marker="o", label=f"{func} ({LABEL[et]})",
                color=COLORS[et], linewidth=2,
                linestyle="-" if et == "gpu_kernel" else "--")

    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "speedup_scaling.png")


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    run_benchmark()

    rows = load_results()
    print(f"\nLoaded {len(rows)} benchmark rows. Generating plots …\n")

    plot_speedup_vs_popsize(rows)
    plot_absolute_times(rows)
    plot_speedup_heatmap(rows)
    plot_eval_type_comparison(rows)
    plot_fitness_quality(rows)
    plot_scaling(rows)

    print(f"\nAll plots saved to {PLOT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
