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
#include <string>

#include "benchmarks/SphereFunction.h"
#include "core/Population.h"
#include "utils/AlgorithmBuilder.h"

int main() {
    const std::string config_path = "config.yaml";

    galib::benchmark::SphereFunction<double> fitness_fn(10, -5.12, 5.12);
    galib::Population<double> population(100, 10);
    population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

    auto algorithm = galib::utils::AlgorithmBuilder<double>::build(config_path, fitness_fn);
    algorithm->run(population);

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

Configuration files are in YAML format and located in the `configs/` directory. Key sections:

- `algorithm.type`: Choose from "standard", "cellular", "island", "differential_evolution"
- `algorithm.max_generations`: Number of generations
- `algorithm.mutation_rate`, `algorithm.crossover_rate`: Genetic operator rates
- `algorithm.pop_size`: Population size for Differential Evolution
- `algorithm.threads`: Number of worker threads. `0` lets the backend choose automatically.
- `algorithm.selection`, `algorithm.mutation`, `algorithm.crossover`: Configure genetic operators
- Algorithm-specific parameters such as `standard.use_cuda`, `cellular.use_local_elitism`, `island.topology`, and `differential_evolution.f_weight`

Notes:

- `Standard GA` uses OpenMP threads.
- `Differential Evolution` uses OpenMP threads.
- `Cellular GA` uses TBB and interprets a regular `Population` as a 2D grid.
- For `Cellular GA`, the grid shape is inferred automatically from `population.size()`.
- `algorithm.pop_size` is used by Differential Evolution configs, while Standard GA and Cellular GA examples create the population in code.

The installed sample config is:

- `share/GALib/configs/full_config_example.yaml`

Inside the repository you can also use:

- `configs/full_config_example.yaml`
- `configs/other/config_standard.yaml`
- `configs/other/config_de.yaml`
- `configs/other/config_island.yaml`

## Examples

- `standard_ga_example.cpp`: Basic generational GA
- `cellular_ga_example.cpp`: Grid-based cellular GA
- `island_ga_example.cpp`: Multi-population island model with MPI
- `differential_evolution_example.cpp`: DE algorithm

Each example loads a YAML config and runs the optimization, printing the best fitness found.
