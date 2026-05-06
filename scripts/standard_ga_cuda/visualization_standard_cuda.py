"""
Standard GA CUDA — Population Evolution Visualizer
----------------------------------------------------
Runs the standard_ga_cuda_example binary with the given config, then produces
an animated 2-D contour + 3-D surface plot of how the population evolves.

Usage (from project root):
    python3 scripts/standard_ga_cuda/visualization_standard_cuda.py [config_path]

Default config: configs/config_standard_cuda_viz.yaml
"""

import subprocess
import sys
import os
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    from benchmarks import FUNCTIONS_MAP
except ImportError:
    sys.exit("[ERROR] scripts/benchmarks.py not found. Run from project root.")

BIN = PROJECT_ROOT / "build" / "standard_ga_cuda_example"

# ── config ────────────────────────────────────────────────────────────────────
config_arg = sys.argv[1] if len(sys.argv) > 1 else "configs/config_standard_cuda_viz.yaml"

if os.path.isabs(config_arg):
    CONFIG_PATH = Path(config_arg)
elif config_arg.startswith("configs/"):
    CONFIG_PATH = PROJECT_ROOT / config_arg
else:
    CONFIG_PATH = PROJECT_ROOT / "configs" / config_arg

if not CONFIG_PATH.exists():
    sys.exit(f"[ERROR] Config not found: {CONFIG_PATH}")

# Parse config manually (avoid yaml dependency)
import re
def _read_yaml_value(text, key):
    m = re.search(rf'^\s*{key}\s*:\s*"?([^"\n#]+)"?', text, re.MULTILINE)
    return m.group(1).strip() if m else None

config_text = CONFIG_PATH.read_text()
prob_name = _read_yaml_value(config_text, "name") or "Rastrigin"
lb        = float(_read_yaml_value(config_text, "lower_bound") or -5.12)
ub        = float(_read_yaml_value(config_text, "upper_bound") or  5.12)
log_path_rel = _read_yaml_value(config_text, "path") or "logs/cuda_evolution.csv"

CSV_PATH = PROJECT_ROOT / log_path_rel

# ── run the binary ────────────────────────────────────────────────────────────
if not BIN.exists():
    sys.exit(f"[ERROR] Binary not found: {BIN}\n"
             "  Build first:  cmake --build build --target standard_ga_cuda_example")

print(f"Running {BIN.name} with {CONFIG_PATH.name} …")
result = subprocess.run([str(BIN), str(CONFIG_PATH)], capture_output=True, text=True,
                        cwd=str(PROJECT_ROOT))
print(result.stdout.strip())
if result.returncode != 0:
    sys.exit(f"[ERROR] Binary failed:\n{result.stderr}")

if not CSV_PATH.exists():
    sys.exit(f"[ERROR] Log file not produced: {CSV_PATH}\n"
             "  Make sure output.file.enabled=true in your config.")

# ── load CSV ──────────────────────────────────────────────────────────────────
print(f"Loading log: {CSV_PATH.relative_to(PROJECT_ROOT)}")
rows = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            "generation": int(r["generation"]),
            "x": float(r["gene_0"]),
            "y": float(r["gene_1"]),
            "fitness": float(r["fitness"]),
        })

generations = sorted({r["generation"] for r in rows})
print(f"  {len(generations)} generations, {len(rows)} total individuals")

# ── landscape ─────────────────────────────────────────────────────────────────
if prob_name not in FUNCTIONS_MAP:
    sys.exit(f"[ERROR] Unknown function '{prob_name}'. "
             f"Available: {list(FUNCTIONS_MAP.keys())}")

target_fn = FUNCTIONS_MAP[prob_name]
res = 120
xs = np.linspace(lb, ub, res)
ys = np.linspace(lb, ub, res)
X, Y = np.meshgrid(xs, ys)
Z = target_fn(X, Y)

# ── figure setup ──────────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 8))

ax3d = fig.add_subplot(121, projection="3d")
ax2d = fig.add_subplot(122)

ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.3, antialiased=True)
ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Fitness")
scatter3d = ax3d.scatter([], [], [], c="red", s=15, edgecolors="black", zorder=5)

contour = ax2d.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.8)
fig.colorbar(contour, ax=ax2d, label="Fitness")
scatter2d = ax2d.scatter([], [], c="red", s=20, edgecolors="black", zorder=5)

ax2d.set_aspect("equal")
ax2d.set_xlabel("X"); ax2d.set_ylabel("Y")
ax2d.set_xlim(lb, ub); ax2d.set_ylim(lb, ub)
ax2d.axhline(0, color="white", linewidth=0.8, alpha=0.4)
ax2d.axvline(0, color="white", linewidth=0.8, alpha=0.4)
ax2d.grid(True, color="gray", linestyle="--", alpha=0.3)

# best fitness line chart overlay (inset)
ax_fit = ax2d.inset_axes([0.02, 0.72, 0.40, 0.26])
ax_fit.set_facecolor((0, 0, 0, 0.5))
ax_fit.tick_params(labelsize=6, colors="white")
ax_fit.set_xlabel("Generation", fontsize=6, color="white")
ax_fit.set_ylabel("Best fitness", fontsize=6, color="white")
best_per_gen = [min(r["fitness"] for r in rows if r["generation"] == g) for g in generations]
ax_fit.plot(generations, best_per_gen, color="cyan", linewidth=1)
fit_marker, = ax_fit.plot([], [], "ro", markersize=4)

# ── animation ─────────────────────────────────────────────────────────────────
def update(gen):
    curr = [r for r in rows if r["generation"] == gen]
    if not curr:
        return scatter3d, scatter2d, fit_marker

    px = np.array([r["x"] for r in curr])
    py = np.array([r["y"] for r in curr])
    pz = target_fn(px, py)

    scatter3d._offsets3d = (px, py, pz)
    scatter2d.set_offsets(np.c_[px, py])

    idx = generations.index(gen)
    fit_marker.set_data([gen], [best_per_gen[idx]])

    fig.suptitle(
        f"Standard GA CUDA — {prob_name}  |  Generation {gen}  |  "
        f"Best fitness: {best_per_gen[idx]:.4f}",
        fontsize=14
    )
    return scatter3d, scatter2d, fit_marker

ani = FuncAnimation(fig, update, frames=generations, interval=80, blit=False)

# ── output dirs ───────────────────────────────────────────────────────────────
IMG_DIR = PROJECT_ROOT / "visualizations" / "images"
ANI_DIR = PROJECT_ROOT / "visualizations" / "animations"
IMG_DIR.mkdir(parents=True, exist_ok=True)
ANI_DIR.mkdir(parents=True, exist_ok=True)

tag = f"standard_cuda_{prob_name.lower()}"

static_path = IMG_DIR / f"plot_{tag}.png"
update(generations[-1])
fig.savefig(static_path, dpi=150, bbox_inches="tight")
print(f"Static plot:  {static_path.relative_to(PROJECT_ROOT)}")

mp4_path = ANI_DIR / f"{tag}.mp4"
gif_path = ANI_DIR / f"{tag}.gif"

try:
    ani.save(str(mp4_path), writer="ffmpeg", fps=10, dpi=150)
    print(f"MP4 saved:    {mp4_path.relative_to(PROJECT_ROOT)}")
except Exception as e:
    print(f"MP4 skipped:  {e}")

try:
    ani.save(str(gif_path), writer="pillow", fps=10)
    print(f"GIF saved:    {gif_path.relative_to(PROJECT_ROOT)}")
except Exception as e:
    print(f"GIF skipped:  {e}")

print("Done.")
