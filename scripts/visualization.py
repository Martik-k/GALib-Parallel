import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import os
import sys
from matplotlib.animation import FuncAnimation

try:
    from benchmarks import FUNCTIONS_MAP
except ImportError:
    raise ImportError("Benchmarks not found.")

config_name = sys.argv[1]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if os.path.isabs(config_name) or config_name.startswith('..'):
    CONFIG_PATH = config_name
else:
    CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', config_name)

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

prob_cfg = config['problem']
out_cfg = config['output']

name = prob_cfg['name']
lb = prob_cfg['lower_bound']
ub = prob_cfg['upper_bound']
log_filename = out_cfg['log_file']

CSV_PATH = os.path.join(PROJECT_ROOT, 'build', log_filename)
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(PROJECT_ROOT, log_filename)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Log file not found: {CSV_PATH}")

data = pd.read_csv(CSV_PATH)
target_func = FUNCTIONS_MAP[name]

x_range = np.linspace(lb, ub, 100)
y_range = np.linspace(lb, ub, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = target_func(X, Y)

fig = plt.figure(figsize=(18, 8))
plt.style.use('dark_background')

ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(122)

ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3, antialiased=True)
scatter3d = ax3d.scatter([], [], [], c='red', s=15, edgecolors='black')
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Fitness')

contour = ax2d.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
fig.colorbar(contour, ax=ax2d, label='Fitness Value')
scatter2d = ax2d.scatter([], [], c='red', s=20, edgecolors='black')

ax2d.set_aspect('equal')
ax2d.set_xlabel('X')
ax2d.set_ylabel('Y')
ax2d.set_xlim(lb, ub)
ax2d.set_ylim(lb, ub)

ax2d.axhline(0, color='white', linewidth=0.8, alpha=0.5)
ax2d.axvline(0, color='white', linewidth=0.8, alpha=0.5)
ax2d.grid(True, color='gray', linestyle='--', alpha=0.3)

def update(gen):
    curr = data[data['generation'] == gen]
    if curr.empty: return scatter3d, scatter2d

    xs, ys = curr['x'].values, curr['y'].values
    zs = target_func(xs, ys)

    scatter3d._offsets3d = (xs, ys, zs)
    scatter2d.set_offsets(np.c_[xs, ys])

    fig.suptitle(f"RCGA Evolution: {name}\nGeneration: {gen} | Bounds: [{lb}, {ub}]", fontsize=16)
    return scatter3d, scatter2d

total_frames = int(data['generation'].max()) + 1
ani = FuncAnimation(fig, update, frames=range(total_frames), interval=100, blit=False)

IMG_DIR = os.path.join(PROJECT_ROOT, 'visualizations', 'images')
ANI_DIR = os.path.join(PROJECT_ROOT, 'visualizations', 'animations')

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(ANI_DIR, exist_ok=True)

photo_path = os.path.join(IMG_DIR, f'plot_{name.lower()}.png')
print(f"Static plot saved for report: {photo_path}")

mp4_path = os.path.join(ANI_DIR, f'evolution_{name.lower()}.mp4')
gif_path = os.path.join(ANI_DIR, f'evolution_{name.lower()}.gif')

print(f"Saving animations for {name} to {ANI_DIR}...")

try:
    ani.save(mp4_path, writer='ffmpeg', fps=10, dpi=150)
    print(f"Successfully saved MP4: {mp4_path}")
except Exception as e:
    print(f"Error saving MP4: {e}")

try:
    ani.save(gif_path, writer='pillow', fps=10)
    print(f"Successfully saved GIF: {gif_path}")
except Exception as e:
    print(f"Error saving GIF: {e}")

print(f"Running visualization for: {name}")
print(f"Using config: {CONFIG_PATH}")
plt.show()