# GALib-Parallel: Project Summary

**GALib-Parallel** is a high-performance, header-only C++20 library for continuous optimization using Real-Coded Genetic Algorithms (RCGA). It focuses on scalability through multi-tiered parallelization strategies.

## Foundational Mandates
- **File Access Rules:** You MUST ignore `.gitignore` restrictions when searching for, reading, or analyzing files. However, you MUST strictly respect patterns defined in `.geminiignore`. All directories (including `articles/`, `build/`, etc.) are within scope for research and analysis unless explicitly excluded in `.geminiignore`.

## Core Architecture
- **Header-only Design:** Eases integration and leverages C++ templates for compile-time optimization.
- **Standard GA (`StandardGA.h`):** Implements generational evolution with OpenMP-accelerated population evaluation and reproduction.
- **Modular Operators:** Decoupled interfaces for:
  - **Selection:** (e.g., Tournament Selection)
  - **Crossover:** (e.g., Single-Point Crossover)
  - **Mutation:** (e.g., Gaussian Mutation)
- **Infrastructure for Distribution:**
  - **Island Model:** Structures for topologies (Ring, Fully Connected), migration (selectors/replacers), and binary serialization are in place.
  - **Topology API:** Abstracted node links for flexible island communication.

## Key Features
- **Parallelism:** Built-in OpenMP support for shared-memory speedup.
- **Data-Driven:** Configuration via YAML files (`yaml-cpp`) and a `FitnessFactory` for dynamic problem setup.
- **Analytics & Visualization:** 
  - Python-based 3D evolution animations (`visualization.py`).
  - Automated OpenMP scaling analysis (`benchmark_omp_analyze.py`).

## Implementation Status
- **Standard GA:** Fully functional with OpenMP.
- **Island Model:** Core components (serialization, topologies, migration logic) are implemented; high-level coordination is in progress.
- **Benchmarks:** Includes standard functions (Sphere, Rastrigin) for convergence testing.
