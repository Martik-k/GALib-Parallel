# GALib Benchmarks

This directory contains tools for benchmarking GALib algorithms, particularly measuring execution time vs number of threads.

## Files

- `benchmark_runner.cpp`: C++ executable for running benchmarks with different thread counts.
- `benchmark_plot.py`: Python script to plot benchmark results.

## Usage

1. Build the benchmark runner:
   ```
   make benchmark_runner
   ```

2. Run benchmarks:
   ```
   ./build/benchmark_runner --threads 1,2,4,8 --config configs/full_config_example.yaml --output results.csv
   ```

3. Plot results:
   ```
   python benchmarks/benchmark_plot.py results.csv
   ```

## Options

- `--threads`: Comma-separated list of thread counts (e.g., 1,2,4,8)
- `--config`: Path to YAML config file
- `--output`: Output CSV file (default: benchmark_results.csv)
- `--num_genes`: Number of genes (default: 10)
- `--pop_size`: Population size (default: 50)
- `--runs`: Number of runs per thread count for averaging (default: 5)