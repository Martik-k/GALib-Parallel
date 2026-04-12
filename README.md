# GALib-Parallel

A header-only C++ library for continuous optimization using a Real-Coded Genetic Algorithm (RCGA), accelerated with OpenMP. The project includes Python tools for 3D visualization of the optimization process and performance benchmarking.

## Key Features

* **Header-only Architecture:** Easy integration into existing C++ projects without complex linking.
* **OpenMP Parallelization:** Accelerated population evaluation and reproduction steps.
* **Data-Driven Configuration:** YAML-based setup for algorithm parameters, functions, and bounds.
* **Continuous Search Space:** Implements RCGA natively with double precision floating-point representation.
* **Python Analytics Tooling:** Automated scripts for generating 2D/3D evolution animations and parallel speedup graphs.

## Prerequisites

**C++ Core:**
* C++20 compatible compiler (GCC/Clang)
* CMake (>= 3.15)
* OpenMP

**Python Analytics:**
* Python 3.x
* Python packages: `numpy`, `matplotlib`, `pandas`, `pyyaml`
* System packages: `ffmpeg` (for MP4 animation generation)

## Project Structure

* `include/` - Core header-only C++ library files.
* `examples/` - Example source files (`main.cpp`) utilizing the library.
* `configs/` - YAML configuration files for benchmarking (e.g., Sphere, Rastrigin).
* `scripts/` - Python scripts for data visualization and performance testing.
* `build/` - Compilation output and generated CSV logs.

## Build Instructions

1. Clone the repository and navigate to the project root:
   ```bash
   git clone <repository-url>
   cd GALib-Parallel
   ```

2. Create a build directory and compile the project:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage

### 1. Running the Genetic Algorithm
Execute the compiled binary from the build directory, passing the path to the desired YAML configuration file:

```bash
cd build
./ga_example ../configs/viz_rastrigin.yaml  # or viz_sphere.yaml
```

To run the CUDA backend (when CUDA toolkit is available during CMake configure), use a config with:

```yaml
algorithm:
   backend: "CUDA"
```

Example:

```bash
./ga_example ../configs/viz_rastrigin_cuda.yaml
```

This runs the algorithm and generates a CSV log file containing the population history.

### 2. 3D Evolution Visualization
Generates MP4 and GIF animations showing the population converging toward the global minimum, along with a static PNG plot.

```bash
# Run from the project root
python3 scripts/visualization.py viz_rastrigin.yaml  # or viz_sphere.yaml
```

Files are automatically saved to visualizations/animations/ and visualizations/images/

### 3. OpenMP Performance Benchmarking
Runs the algorithm across multiple thread counts to test parallel efficiency and generates execution time and speedup graphs.

```bash
# Run from the project root
python3 scripts/benchmark_omp.py
```

The resulting plot is saved to visualizations/images/omp_performance_results.png.

