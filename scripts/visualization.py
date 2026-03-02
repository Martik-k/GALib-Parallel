import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import os

def sphere(x, y):
    return x**2 + y**2

def rastrigin(x, y, A=10):
    return A*2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

csv_path = '../build/evolution_history.csv'
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit()

data = pd.read_csv(csv_path)
num_gens = int(data['generation'].max())
x_range = np.linspace(-5.12, 5.12, 100)
y_range = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x_range, y_range)

benchmarks = [
    {"name": "sphere", "func": sphere},
    {"name": "rastrigin", "func": rastrigin}
]

for bench in benchmarks:
    name = bench["name"]
    func = bench["func"]
    Z = func(X, Y)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
    scatter = ax.scatter([], [], c='red', s=10, label='Individuals')
    ax.set_title(f'Evolution: {name.capitalize()}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    def update(gen, scatter, data, title_ax, name):
        current_gen = data[data['generation'] == gen]
        scatter.set_offsets(np.c_[current_gen['x'], current_gen['y']])
        title_ax.set_title(f'Evolution: {name.capitalize()} | Generation: {gen}')
        return scatter,

    ani = FuncAnimation(
        fig, update, frames=range(0, num_gens + 1),
        fargs=(scatter, data, ax, name),
        blit=True, interval=400
    )

    print(f"Processing {name}...")

    try:
        ani.save(f'evolution_{name}.mp4', writer='ffmpeg', fps=10)
        print(f"  - {name}.mp4 saved")
    except Exception as e:
        print(f"  - Could not save MP4 for {name}: {e}")

    ani.save(f'evolution_{name}.gif', writer='pillow', fps=10)
    print(f"  - {name}.gif saved")

    plt.close(fig)

print("\nDone! All visualizations generated.")