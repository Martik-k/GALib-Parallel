#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import glob

def plot_time_benchmark(df, output_dir, prefix):
    # Filter out invalid threads and convert to int
    df = df[df['threads'] > 0].copy()
    df['threads'] = df['threads'].astype(int)
    
    algorithms = df['algorithm'].unique()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for algo in algorithms:
        algo_df = df[df['algorithm'] == algo]

        # Time vs Threads
        axes[0].plot(algo_df['threads'], algo_df['time_seconds'], marker='o', linestyle='-', label=algo)
        axes[0].set_xlabel('Number of Threads')
        axes[0].set_ylabel('Execution Time (seconds)')
        axes[0].set_title('Execution Time vs Threads')
        axes[0].grid(True)
        axes[0].legend()
        axes[0].set_xticks(sorted(algo_df['threads'].unique()))

        # Speedup vs Threads
        axes[1].plot(algo_df['threads'], algo_df['speedup'], marker='o', linestyle='-', label=algo)
        axes[1].set_xlabel('Number of Threads')
        axes[1].set_ylabel('Speedup')
        axes[1].set_title('Speedup vs Threads')
        axes[1].grid(True)
        axes[1].legend()
        axes[1].set_xticks(sorted(algo_df['threads'].unique()))

        # Efficiency vs Threads
        axes[2].plot(algo_df['threads'], algo_df['efficiency'], marker='o', linestyle='-', label=algo)
        axes[2].set_xlabel('Number of Threads')
        axes[2].set_ylabel('Efficiency')
        axes[2].set_title('Efficiency vs Threads')
        axes[2].grid(True)
        axes[2].legend()
        axes[2].set_xticks(sorted(algo_df['threads'].unique()))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_benchmark.png'))
    print(f"Time benchmark plot saved to {os.path.join(output_dir, f'{prefix}_benchmark.png')}")
    plt.close()

def plot_quality_benchmark(df, output_dir, prefix):
    # Only convergence plot for quality benchmark, find all convergence files
    import glob
    conv_files = glob.glob(os.path.join(output_dir, 'quality_convergence_*.csv'))
    for conv_file in conv_files:
        algo = os.path.basename(conv_file).replace('quality_convergence_', '').replace('.csv', '')
        conv_df = pd.read_csv(conv_file)
        plt.figure(figsize=(10, 6))
        plt.plot(conv_df['generation'], conv_df['fitness'], marker='o', linestyle='-', label=f'{algo} (distance to optimum)')
        plt.xlabel('Generation')
        plt.ylabel('Distance to Optimum (Best Fitness)')
        plt.title(f'Quality Convergence: Distance to Optimum vs Generation ({algo})')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{prefix}_quality_convergence_{algo}.png'))
        print(f"Quality convergence plot for {algo} saved to {os.path.join(output_dir, f'{prefix}_quality_convergence_{algo}.png')}")
        plt.close()

def plot_convergence(df, output_dir, prefix):
    # Add missing columns if not present (for quality convergence files)
    if 'threads' not in df.columns:
        df['threads'] = 1
    if 'algorithm' not in df.columns:
        # Extract algorithm from prefix, e.g., 'quality_convergence_standard' -> 'standard'
        parts = prefix.split('_')
        if len(parts) >= 3 and parts[0] == 'quality' and parts[1] == 'convergence':
            df['algorithm'] = parts[2]
        else:
            df['algorithm'] = 'unknown'
    
    # For files with individual fitness, group by generation and take best (min) fitness
    if 'fitness' in df.columns and 'best_fitness' not in df.columns:
        df = df.groupby('generation')['fitness'].min().reset_index()
        df.rename(columns={'fitness': 'best_fitness'}, inplace=True)
        df['threads'] = 1  # Ensure threads is set
        df['algorithm'] = df['algorithm'].iloc[0] if 'algorithm' in df.columns else 'unknown'
    
    # Convert columns to appropriate types
    df['threads'] = df['threads'].astype(int)
    df['generation'] = df['generation'].astype(int)
    
    algorithms = df['algorithm'].unique()
    threads_list = df['threads'].unique()

    for algo in algorithms:
        algo_df = df[df['algorithm'] == algo]
        plt.figure(figsize=(10, 6))

        for threads in threads_list:
            thread_df = algo_df[algo_df['threads'] == threads]
            if not thread_df.empty:
                thread_df = thread_df.sort_values('generation')
                plt.plot(thread_df['generation'], thread_df['best_fitness'], linestyle='-', marker=None, label=f'{algo} ({threads} threads)')

        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title(f'Quality Convergence: Best Fitness vs Generation ({algo})')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{prefix}_{algo}.png'))
        print(f"Quality convergence plot for {algo} saved to {os.path.join(output_dir, f'{prefix}_{algo}.png')}")
        plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_plot.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_dir = 'results'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract prefix from csv_file name (e.g., 'de_time' from 'results/de_time.csv')
    base_name = os.path.basename(csv_file)
    prefix = base_name.replace('.csv', '').replace('results/', '')

    # Read CSV if it exists, otherwise assume quality benchmark
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        df = pd.read_csv(csv_file)
        # Determine benchmark type
        if 'generation' in df.columns:
            plot_convergence(df, output_dir, prefix)
        elif 'time_seconds' in df.columns:
            plot_time_benchmark(df, output_dir, prefix)
        elif 'best_fitness' in df.columns:
            plot_quality_benchmark(df, output_dir, prefix)
        else:
            print("Unknown benchmark type in CSV file")
            sys.exit(1)
    else:
        # Assume quality benchmark without CSV
        plot_quality_benchmark(None, output_dir, prefix)

if __name__ == "__main__":
    main()