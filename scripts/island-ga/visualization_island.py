import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add scripts directory to sys.path to allow importing benchmarks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from benchmarks import FUNCTIONS_MAP
except ImportError:
    # Fallback for different execution contexts
    sys.path.append(os.path.join(os.getcwd(), 'scripts'))
    from benchmarks import FUNCTIONS_MAP

class IslandData:
    """Represents the evolutionary data for a single GA island rank."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.rank = self._extract_rank(file_path)
        self.df = pd.read_csv(file_path)
        self.max_generation = self.df['generation'].max()

    def _extract_rank(self, file_path):
        match = re.search(r'rank_(\d+)', os.path.basename(file_path))
        return int(match.group(1)) if match else -1

    def get_generation_subset(self, gen):
        return self.df[self.df['generation'] == gen]

class IslandVisualizer:
    """Orchestrates the grid-based animation of all GA islands."""
    def __init__(self, logs_dir, function_name, bounds, save_path, migrant_generations=5, interval=100, frame_step=1):
        self.logs_dir = logs_dir
        self.function_name = function_name
        self.target_func = FUNCTIONS_MAP[function_name]
        self.bounds = bounds
        self.save_path = save_path
        self.migrant_generations = migrant_generations
        self.interval = interval
        self.frame_step = frame_step
        self.islands = self._load_islands()
        self.max_gen = max(island.max_generation for island in self.islands)

        self._detect_migrations()

        # Setup grid layout
        num_islands = len(self.islands)
        self.cols = int(np.ceil(np.sqrt(num_islands)))
        self.rows = int(np.ceil(num_islands / self.cols))

        # Prepare figure
        plt.style.use('default')
        self.fig, self.axes = plt.subplots(self.rows, self.cols, figsize=(self.cols * 5, self.rows * 5), squeeze=False)
        self.axes = self.axes.flatten()
        self.scatters = []

        self._prepare_backgrounds()

    def _load_islands(self):
        files = [os.path.join(self.logs_dir, f) for f in os.listdir(self.logs_dir) if f.endswith('.csv') and 'rank' in f]
        islands = [IslandData(f) for f in files]
        return sorted(islands, key=lambda x: x.rank)

    def _detect_migrations(self):
        """Identifies migrants by comparing genotypes across islands and generations."""
        if self.migrant_generations <= 0:
            return

        print(f"Detecting migrations (persistence: {self.migrant_generations} generations)...")
        gene_cols = [c for c in self.islands[0].df.columns if c.startswith('gene_')]

        for island in self.islands:
            island.df['is_migrant_display'] = False

        # rank -> {genotype_tuple: age}
        prev_gen_data = {}

        for island in self.islands:
            gen0 = island.df[island.df['generation'] == 0]
            prev_gen_data[island.rank] = {tuple(row[gene_cols]): -1 for _, row in gen0.iterrows()}

        for gen in range(1, self.max_gen + 1):
            current_gen_data = {}
            all_prev_genotypes = {} # genotype -> rank
            for r, data in prev_gen_data.items():
                for g_tuple in data:
                    all_prev_genotypes[g_tuple] = r

            for island in self.islands:
                mask = island.df['generation'] == gen
                subset_indices = island.df.index[mask]

                island_prev_genotypes = prev_gen_data[island.rank]
                current_island_genotypes = {}

                for idx in subset_indices:
                    row = island.df.loc[idx]
                    g_tuple = tuple(row[gene_cols])

                    age = -1
                    if g_tuple in all_prev_genotypes and all_prev_genotypes[g_tuple] != island.rank:
                        age = 0
                    elif g_tuple in island_prev_genotypes and island_prev_genotypes[g_tuple] != -1:
                        age = island_prev_genotypes[g_tuple] + 1

                    if 0 <= age < self.migrant_generations:
                        island.df.at[idx, 'is_migrant_display'] = True
                        current_island_genotypes[g_tuple] = age
                    else:
                        current_island_genotypes[g_tuple] = -1

                current_gen_data[island.rank] = current_island_genotypes
            prev_gen_data = current_gen_data

    def _prepare_backgrounds(self):
        """Pre-calculates and plots the function landscape for all subplots."""
        x = np.linspace(self.bounds[0], self.bounds[1], 100)
        y = np.linspace(self.bounds[0], self.bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = self.target_func(X, Y)

        for i, ax in enumerate(self.axes):
            if i < len(self.islands):
                island = self.islands[i]
                ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
                ax.set_title(f"Island Rank {island.rank}", fontsize=12)
                ax.set_xlim(self.bounds)
                ax.set_ylim(self.bounds)

                # Initialize with a solid color to ensure markers are filled
                scatter = ax.scatter([], [], c='red', s=15, edgecolors='black', alpha=0.8)
                self.scatters.append(scatter)
            else:
                ax.axis('off')

    def update(self, gen):
        """Animation update function for a specific generation."""
        changed_artists = []
        for i, island in enumerate(self.islands):
            subset = island.get_generation_subset(gen)
            if not subset.empty:
                coords = np.c_[subset['gene_0'].values, subset['gene_1'].values]
                self.scatters[i].set_offsets(coords)

                # Color coding: Blue for migrants, Red for natives
                colors = ['blue' if m else 'red' for m in subset.get('is_migrant_display', [False]*len(subset))]
                self.scatters[i].set_facecolors(colors)
                self.scatters[i].set_edgecolors('black')

                changed_artists.append(self.scatters[i])

        self.fig.suptitle(f"Island Model Evolution: {self.function_name} | Generation {gen}", fontsize=16)
        return changed_artists

    def animate(self):
        """Creates and saves the evolution animation."""
        # Ensure save_path has an extension for ffmpeg/pillow
        if not os.path.splitext(self.save_path)[1]:
            self.save_path += ".gif"

        print(f"Creating animation for {len(self.islands)} islands up to generation {self.max_gen} (step: {self.frame_step})...")
        ani = FuncAnimation(
            self.fig, self.update, frames=range(0, self.max_gen + 1, self.frame_step),
            blit=False, interval=self.interval
        )

        print(f"Saving animation to {self.save_path}...")
        fps = 1000 // self.interval
        if self.save_path.lower().endswith('.gif'):
            ani.save(self.save_path, writer='pillow', fps=fps)
        else:
            # mp4 or other video formats
            ani.save(self.save_path, writer='ffmpeg', fps=fps, dpi=100)
        print(f"Successfully saved animation to {self.save_path}")

def main():
    parser = argparse.ArgumentParser(description="Animate Island Model GA evolution across multiple ranks.")
    parser.add_argument("--logs_dir", required=True, help="Directory containing island CSV logs.")
    parser.add_argument("--function", default="Rastrigin", choices=list(FUNCTIONS_MAP.keys()), help="Target function name.")
    parser.add_argument("--save_path", default="evolution.gif", help="Path to save the animation (e.g., evolution.gif or .mp4).")
    parser.add_argument("--save_mp4", action="store_true", help="Save the animation as an MP4 instead of GIF.")
    parser.add_argument("--bounds", type=float, nargs=2, default=[-5.12, 5.12], help="Visualization boundaries (min max).")
    parser.add_argument("--migrant_generations", type=int, default=5, help="Number of generations to highlight migrants (default: 5).")
    parser.add_argument("--interval", type=int, default=100, help="Delay between frames in milliseconds (default: 100).")
    parser.add_argument("--frame_step", type=int, default=1, help="Show every Nth generation in the animation (default: 1).")

    args = parser.parse_args()

    # Check flag to explicitly enforce .mp4 extension
    if args.save_mp4:
        base_path = os.path.splitext(args.save_path)[0]
        args.save_path = f"{base_path}.mp4"

    if not os.path.isdir(args.logs_dir):
        print(f"Error: {args.logs_dir} is not a valid directory.")
        return

    visualizer = IslandVisualizer(
        logs_dir=args.logs_dir,
        function_name=args.function,
        bounds=args.bounds,
        save_path=args.save_path,
        migrant_generations=args.migrant_generations,
        interval=args.interval,
        frame_step=args.frame_step
    )
    visualizer.animate()

if __name__ == "__main__":
    main()