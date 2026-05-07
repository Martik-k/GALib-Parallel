# GALib-Parallel

A header-only C++ library for continuous optimization using various evolutionary algorithms (Standard GA, Cellular GA, Island GA, Differential Evolution), accelerated with OpenMP, MPI, and CUDA. The project includes Python tools for 3D visualization of the optimization process and performance benchmarking.

## Key Features

* **Header-only Architecture:** Easy integration into existing C++ projects without complex linking.
* **Multiple Algorithms:** Support for Standard GA, Cellular GA, Island GA (with MPI), and Differential Evolution.
* **Parallelization:** OpenMP for Standard GA and Differential Evolution, TBB for Cellular GA, MPI for the island model, CUDA for GPU-enabled builds.
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

1. Clone the repository and move to the project root:
   ```bash
   git clone <repository-url>
   cd GALib-Parallel
   ```

2. Configure and build:
   ```bash
   cmake -S . -B build
   cmake --build build
   ```

3. Build a specific example target if needed:
   ```bash
   cmake --build build --target ga_example
   cmake --build build --target cellular_example
   cmake --build build --target de_example
   cmake --build build --target island_example
   ```

`island_example` is only available when MPI is found. `ga_cuda` and `test_ga_cuda` are only available when CUDA is found.

## Installation

### 1. Install Locally

Install into a local folder inside the repository:

```bash
cmake -S . -B build
cmake --build build
cmake --install build --prefix ./install
```

This creates:

* `install/include/` - public headers
* `install/lib/cmake/GALib/` - CMake package files for `find_package(GALib)`
* `install/share/GALib/examples/` - installed example sources
* `install/share/GALib/configs/` - installed sample configs

### 2. Install System-Wide

If you want `find_package(GALib)` to work without extra path flags, install into a standard system prefix:

```bash
cmake -S . -B build
cmake --build build
sudo cmake --install build --prefix /usr/local
```

### 3. CUDA Build

To build and install the CUDA-enabled package variant of GALib:

```bash
cmake -S . -B build-cuda
cmake --build build-cuda
cmake --install build-cuda --prefix ./install-cuda
```

You can control the CUDA architecture set with:

```bash
cmake -S . -B build-cuda -DGALIB_CUDA_ARCHITECTURES=native
```

Examples of explicit architecture values:

```bash
cmake -S . -B build-cuda -DGALIB_CUDA_ARCHITECTURES=86
cmake -S . -B build-cuda -DGALIB_CUDA_ARCHITECTURES="75;86"
```

## Using GALib In Your Own Project

### 1. Quick User Flow

To use GALib in another CMake project, the workflow is:

1. Download the library source code.
2. Build and install the library.
3. In your own project, call `find_package(GALib REQUIRED)`.
4. Link against `GALib::ga_lib`.
5. Optionally link `GALib::ga_cuda` or `GALib::ga_island` if those targets are available.

### 2. Download And Install

Clone the repository and install it either locally or system-wide:

```bash
git clone <repository-url>
cd GALib-Parallel
cmake -S . -B build
cmake --build build
cmake --install build --prefix ./install
```

If you want a system-wide install instead:

```bash
cmake -S . -B build
cmake --build build
sudo cmake --install build --prefix /usr/local
```

If you want the CUDA-enabled package variant in `/usr/local`:

```bash
cmake -S . -B build-cuda
cmake --build build-cuda
sudo cmake --install build-cuda --prefix /usr/local
```

### 3. Add `find_package`

In your own `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyApp LANGUAGES CXX)

find_package(GALib REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE GALib::ga_lib)
```

If GALib was installed into a local prefix such as `./install`, configure your project with:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/absolute/path/to/GALib-Parallel/install
cmake --build build
```

If GALib was installed into `/usr/local`, no extra path is usually needed:

```bash
cmake -S . -B build
cmake --build build
```

### 4. Target Types

GALib exports up to three targets, depending on how it was built:

* `GALib::ga_lib` - base library, always available
* `GALib::ga_cuda` - optional CUDA backend, available only when GALib was built with CUDA support
* `GALib::ga_island` - optional island/MPI backend, available only when GALib was built in an environment with working MPI support

The recommended usage pattern is:

```cmake
find_package(GALib REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE GALib::ga_lib)

if(TARGET GALib::ga_cuda)
    target_link_libraries(my_app PRIVATE GALib::ga_cuda)
    target_compile_definitions(my_app PRIVATE MYAPP_HAS_GALIB_CUDA=1)
endif()

if(TARGET GALib::ga_island)
    target_link_libraries(my_app PRIVATE GALib::ga_island)
endif()
```

### 5. Minimal Example

```cpp
#include "utils/AlgorithmBuilder.h"
#include "utils/FunctionalFitness.h"
#include <iostream>

using namespace galib;

int main() {
    // 1. Define problem
    FunctionalFitness<double> fitness(2, -5.0, 5.0,
        [](const std::vector<double>& p) {
            return p[0] * p[0] + p[1] * p[1];
        }
    );

    // 2. Build algorithm instance from YAML configuration
    auto ga = utils::AlgorithmBuilder<double>::build(
        "configs/config_example.yaml", fitness
    );

    // 3. Evolve the population
    Population<double> pop(100, 2);
    pop.initialize(
        fitness.getLowerBound(),
        fitness.getUpperBound()
    );
    ga->run(pop);

    // 4. Get results
    std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << std::endl;

    return 0;
}

```

### 6. Example User Project Layout

Your own project can look like this:

```text
MyApp/
  CMakeLists.txt
  main.cpp
  config.yaml
```

Then build it like this:

```bash
cmake -S . -B build
cmake --build build
./build/my_app
```

If GALib was installed into a local prefix instead of `/usr/local`, configure with:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/absolute/path/to/GALib-Parallel/install
cmake --build build
./build/my_app
```

## Running The Included Examples

Run the example executables from the project root:

**Standard GA**
```bash
./build/ga_example configs/other/config_standard.yaml
```

**Cellular GA**
```bash
./build/cellular_example configs/full_config_example.yaml
```

**Differential Evolution**
```bash
./build/de_example configs/other/config_de.yaml
```

**Island GA**
```bash
mpirun -np <num_processes> ./build/island_example configs/other/config_island.yaml
```

Each run can generate console output and file logs depending on the selected example and YAML configuration.


## Configuration

Configuration files are in YAML format. The library uses a hierarchical structure to define algorithm parameters, genetic operators, and logging options.

| Parameter | Description | Type | Allowed Values |
| :--- | :--- | :--- | :--- |
| `algorithm.type` | Evolutionary algorithm variant to execute. | String | `standard`, `cellular`, `island`, `differential_evolution` |
| `algorithm.max_generations` | Maximum number of generations (iterations) to run. | Integer | Positive integers |
| `algorithm.mutation_rate` | Probability of mutation for each gene. | Float | `0.0` to `1.0` |
| `algorithm.crossover_rate` | Probability of crossover (used as `CR` in Differential Evolution). | Float | `0.0` to `1.0` |
| `algorithm.use_elitism` | If true, the best individuals are preserved for the next generation. | Boolean | `true`, `false` |
| `algorithm.threads` | Number of worker threads for parallelization (`0` for auto-detection). | Integer | `>= 0` |
| `algorithm.standard.use_cuda` | Enable CUDA-accelerated evaluation (Standard GA only). | Boolean | `true`, `false` |
| `algorithm.differential_evolution.f_weight` | Differential weight factor (Scaling Factor) for DE mutation. | Float | `0.0` to `2.0` |
| `algorithm.cellular.rows` | Number of rows in the grid population (Cellular GA). | Integer | Positive integers |
| `algorithm.cellular.cols` | Number of columns in the grid population (Cellular GA). | Integer | Positive integers |
| `algorithm.cellular.use_local_elitism` | If true, elitism is applied within local neighborhoods. | Boolean | `true`, `false` |
| `algorithm.island.topology` | Communication topology between islands in the archipelago. | String | `fully_connected`, `one_way_ring`, `bidirectional_ring` |
| `algorithm.island.migration_interval` | Number of generations between migration events. | Integer | Positive integers |
| `algorithm.island.migration_size` | Number of individuals sent during each migration event. | Integer | Positive integers |
| `algorithm.island.immigration_quota` | Max fraction of population that can be replaced by immigrants. | Float | `0.0` to `1.0` |
| `algorithm.island.buffer_capacity` | Capacity of the asynchronous migration receiving buffer. | Integer | Positive integers |
| `algorithm.island.replacer.type` | Strategy for integrating incoming migrants. | String | `worst` |
| `algorithm.island.selector.type` | Strategy for selecting individuals for emigration. | String | `elitism` |
| `algorithm.selection.type` | Selection operator type. | String | `tournament`, `best_neighbor` (Cellular GA) |
| `algorithm.selection.tournament_size` | Number of individuals in each tournament. | Integer | `>= 1` |
| `algorithm.mutation.type` | Mutation operator type. | String | `gaussian`, `uniform`, `boundary` |
| `algorithm.mutation.sigma` | Standard deviation for Gaussian mutation. | Float | Positive floats |
| `algorithm.crossover.type` | Crossover operator type. | String | `single_point`, `arithmetic`, `uniform` |
| `output.console.enabled` | Enable logging of progress to the console. | Boolean | `true`, `false` |
| `output.console.interval` | Generations between console status updates. | Integer | Positive integers |
| `output.file.enabled` | Enable logging of evolution data to a CSV file. | Boolean | `true`, `false` |
| `output.file.interval` | Generations between CSV log entries. | Integer | Positive integers |
| `output.file.path` | Path to the output CSV log file. | String | File path string |

The sample configuration are available in the repository: `configs/config_example.yaml`

## Examples

- `standard_ga_example.cpp`: Basic generational GA
- `cellular_ga_example.cpp`: Grid-based cellular GA
- `island_ga_example.cpp`: Multi-population island model with MPI
- `differential_evolution_example.cpp`: DE algorithm

Each example loads a YAML config and runs the optimization, printing the best fitness found.
