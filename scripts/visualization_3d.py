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

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3, antialiased=True)
    scatter = ax.scatter([], [], [], c='red', s=20, depthshade=True, label='Individuals')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')
    ax.legend()

    def update(gen, scatter, data, title_ax, func, name):
        current_gen = data[data['generation'] == gen]
        xs = current_gen['x'].values
        ys = current_gen['y'].values
        zs = func(xs, ys)
        scatter._offsets3d = (xs, ys, zs)
        title_ax.set_title(f'3D Evolution: {name.capitalize()} | Generation: {gen}')
        return scatter,

    ani = FuncAnimation(
        fig, update, frames=range(0, num_gens + 1),
        fargs=(scatter, data, ax, func, name),
        blit=False, interval=400
    )

    print(f"Processing 3D {name}...")

    try:
        ani.save(f'evolution_3d_{name}.mp4', writer='ffmpeg', fps=10)
        print(f"  - 3D {name}.mp4 saved")
    except Exception as e:
        print(f"  - Could not save 3D MP4 for {name}: {e}")

    ani.save(f'evolution_3d_{name}.gif', writer='pillow', fps=10)
    print(f"  - 3D {name}.gif saved")

    plt.close(fig)

print("\nDone! All 3D visualizations generated.")