# GALib-Parallel

A header-only C++ library for continuous optimization using various evolutionary algorithms (Standard GA, Cellular GA, Island GA, Differential Evolution), accelerated with OpenMP, MPI, and CUDA. The project includes Python tools for 3D visualization of the optimization process and performance benchmarking.

## Key Features

* **Header-only Architecture:** Easy integration into existing C++ projects without complex linking.
* **Multiple Algorithms:** Support for Standard GA, Cellular GA, Island GA (with MPI), and Differential Evolution.
* **Parallelization:** OpenMP for intra-population parallelism, MPI for inter-population (island model), CUDA for GPU acceleration.
* **Data-Driven Configuration:** YAML-based setup for algorithm parameters, functions, and bounds.
* **Continuous Search Space:** Implements real-coded algorithms with double precision floating-point representation.
* **Benchmark Functions:** Built-in support for Sphere, Rastrigin, Himmelblau, and De Jong F5 functions.
* **Python Analytics Tooling:** Automated scripts for generating 2D/3D evolution animations and parallel speedup graphs.

## Prerequisites

**C++ Core:**
* C++20 compatible compiler (GCC/Clang)
* CMake (>= 3.15)
* OpenMP
* MPI (for island model, optional)
* CUDA Toolkit (for GPU acceleration, optional)

**Python Analytics:**
* Python 3.x
* Python packages: `numpy`, `matplotlib`, `pandas`, `pyyaml`
* System packages: `ffmpeg` (for MP4 animation generation)

## Project Structure

* `include/` - Core header-only C++ library files.
* `examples/` - Example source files for different algorithms (standard_ga_example.cpp, island_ga_example.cpp, cellular_ga_example.cpp, differential_evolution_example.cpp).
* `configs/` - YAML configuration files for different algorithms and benchmarking (e.g., Sphere, Rastrigin).
* `scripts/` - Python scripts for data visualization (visualization.py, visualization_standard.py, visualization_de.py, visualization_cellular.py, visualization_island.py) and performance testing.
* `build/` - Compilation output and generated CSV logs.
* `src/` - Source files for CUDA implementations.
* `cmake/` - CMake configuration files.

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

   To build specific examples:
   ```bash
   make ga_example          # Standard GA
   make island_example      # Island GA (requires MPI)
   make cellular_example    # Cellular GA
   make de_example          # Differential Evolution
   ```

   Note: Island example requires MPI. If MPI is not found, it won't be built.

## Usage

### 1. Running the Algorithms

The project provides separate executables for each algorithm type. Run them from the project root, passing the path to a YAML configuration file:

**Standard GA:**
```bash
./build/ga_example configs/config_standard.yaml
```

**Island GA (requires MPI):**
```bash
mpirun -np <num_processes> ./build/island_example configs/config_island.yaml
```

**Cellular GA:**
```bash
./build/cellular_example configs/full_config_example.yaml
```

**Differential Evolution:**
```bash
./build/de_example configs/config_de.yaml
```

Each run generates a CSV log file containing the population history for visualization.

### 2. Using GALib in Your Own Project

If you want to use GALib as a C++ library in another project, first build and install it locally:

```bash
cmake -S . -B build
cmake --build build -j4
cmake --install build --prefix ./install
```

This creates a local install tree with:

* `install/include/` - public header files
* `install/lib/cmake/GALib/` - CMake package files for `find_package(GALib)`

In your own project, use the following `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyApp LANGUAGES CXX)

find_package(GALib REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE GALib::ga_lib)
```

Then configure your project with:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/absolute/path/to/GALib-Parallel/install
cmake --build build
```

A minimal `main.cpp` can look like this:

```cpp
#include <yaml-cpp/yaml.h>

#include "benchmarks/HimmelblauFunction.h"
#include "core/GridPopulation.h"
#include "utils/AlgorithmBuilder.h"

int main() {
    YAML::Node config = YAML::LoadFile("config.yaml");

    galib::benchmark::HimmelblauFitness<double> fitness_fn;
    galib::GridPopulation<double> population(10, 10, 2);
    population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

    auto ga = galib::utils::AlgorithmBuilder<double>::buildCellularGA(config, fitness_fn);
    ga->run(population);
}
```


### 3. 3D Evolution Visualization
Generates MP4 and GIF animations showing the population converging toward the global minimum, along with a static PNG plot.

For Standard GA:
```bash
python3 scripts/visualization_standard.py configs/config_standard.yaml
```

For Differential Evolution:
```bash
python3 scripts/visualization_de.py configs/config_de.yaml
```

For Cellular GA:
```bash
python3 scripts/visualization_cellular.py configs/full_config_example.yaml
```

For Island GA:
```bash
python3 scripts/visualization_island.py configs/config_island.yaml
```

Files are automatically saved to `visualizations/animations/` and `visualizations/images/`.

### 4. OpenMP Performance Benchmarking
Runs the algorithm across multiple thread counts to test parallel efficiency and generates execution time and speedup graphs.

```bash
# Run from the project root
python3 scripts/benchmark_omp.py
```

The resulting plot is saved to `visualizations/images/omp_performance_results.png`.

## Configuration

Configuration files are in YAML format and located in the `configs/` directory. Key sections:

- `algorithm.type`: Choose from "standard", "cellular", "island", "differential_evolution"
- `algorithm.max_generations`: Number of generations
- `algorithm.mutation_rate`, `algorithm.crossover_rate`: Genetic operator rates
- `algorithm.pop_size`: Population size (for standard and DE)
- `algorithm.selection`, `algorithm.mutation`, `algorithm.crossover`: Configure genetic operators
- Algorithm-specific parameters (e.g., `cellular.rows`, `island.topology`, `differential_evolution.f_weight`)

See `configs/full_config_example.yaml` for all available options and examples.
- Algorithm-specific sections (e.g., `cellular.rows`, `island.topology`)

See `configs/full_config_example.yaml` for all available options.

## Examples

- `standard_ga_example.cpp`: Basic generational GA
- `cellular_ga_example.cpp`: Grid-based cellular GA
- `island_ga_example.cpp`: Multi-population island model with MPI
- `differential_evolution_example.cpp`: DE algorithm

Each example loads a YAML config and runs the optimization, printing the best fitness found.

