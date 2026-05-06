# Contributing to GALib-Parallel

## Documentation Standard

All public headers must be documented with **Doxygen** comments. This gives the team
consistent in-editor hover-docs (CLion, VS Code) and a browsable HTML reference that
can be generated locally or in CI.

---

## Quick Start

### Install Doxygen + Graphviz

```bash
# Ubuntu / Debian
sudo apt install doxygen graphviz

# macOS
brew install doxygen graphviz
```

### Generate the docs

```bash
# From the project root — plain Doxygen
doxygen Doxyfile

# OR via CMake (if you already have a build directory)
cmake --build build --target docs
```

The output lands in `docs/html/`. Open `docs/html/index.html` in any browser.

---

## What to Document

| Scope | Required |
|---|---|
| Every **public class or struct** | Yes — `@brief` + `@tparam` if templated |
| Every **public method** | Yes — `@brief`, `@param`, `@return` |
| Every **pure-virtual interface method** | Yes — defines the contract |
| Private members / private helpers | No |
| `.cu` implementation details | No |

---

## Comment Style Reference

Use the `/** ... */` block style. One blank `*` line separates the brief from the rest.

### Class / Struct

```cpp
/**
 * @brief One-sentence description of what this class represents or does.
 *
 * Optional longer explanation of design decisions, usage context,
 * or non-obvious constraints.
 *
 * @tparam GeneType The numeric type of each gene (default: double).
 */
template <typename GeneType = double>
class MyClass : public BaseClass<GeneType> {
```

### Method

```cpp
/**
 * @brief Brief description of what the method does.
 *
 * @param population The population to select from. Must not be empty.
 * @param rate       Probability in [0.0, 1.0] that an individual is selected.
 * @return Reference to the selected individual.
 * @throws std::invalid_argument if population is empty.
 */
const Individual<GeneType>& select(const Population<GeneType>& population, double rate);
```

### Pure-Virtual Interface Method

Document the **contract** (what it must do), not the implementation:

```cpp
/**
 * @brief Evaluates the fitness of a candidate solution.
 *
 * Lower return values indicate better fitness (minimisation convention).
 *
 * @param phenotype Gene values of the individual to evaluate.
 * @return Fitness score. Must be finite and deterministic for the same input.
 */
virtual double evaluate(const std::vector<GeneType>& phenotype) const = 0;
```

### Constructor

```cpp
/**
 * @brief Constructs a tournament selector.
 * @param tournament_size Number of individuals sampled per selection. Must be >= 1.
 * @throws std::invalid_argument if tournament_size is 0.
 */
explicit TournamentSelection(std::size_t tournament_size);
```

### `@note` and `@warning`

Use `@note` for helpful hints and `@warning` for things that will silently break:

```cpp
/**
 * @brief Runs the optimisation loop on the given population.
 *
 * @param population In/out: the population to evolve. Modified in place.
 * @note  Population must be initialised before calling run().
 * @warning Not thread-safe. Create one algorithm instance per thread.
 */
void run(Population<GeneType>& population) override;
```

---

## Real Examples from This Codebase

### Good — `FunctionalFitness` (use this as a template)

```cpp
/**
 * @brief A wrapper for FitnessFunction that allows using a lambda or std::function.
 *
 * Eliminates the need to subclass FitnessFunction for every new problem.
 * Supports both uniform bounds (same for all dimensions) and per-dimension bounds.
 *
 * @tparam GeneType The type of the genes (default: double).
 */
template <typename GeneType = double>
class FunctionalFitness : public FitnessFunction<GeneType> {
public:
    /**
     * @brief Constructor for uniform bounds.
     * @param dims   Number of dimensions.
     * @param lower  Lower bound applied to all dimensions.
     * @param upper  Upper bound applied to all dimensions.
     * @param func   Evaluation callable (lambda, function pointer, std::function).
     */
    FunctionalFitness(std::size_t dims, GeneType lower, GeneType upper, EvalFunc func);
```

### Before / After — `TournamentSelection`

**Before (no docs):**
```cpp
template <typename GeneType>
class TournamentSelection : public Selection<GeneType> {
    explicit TournamentSelection(size_t tournament_size) { ... }
    const Individual<GeneType>& select(const Population<GeneType>& population) override;
};
```

**After (documented):**
```cpp
/**
 * @brief Selects individuals via k-way tournament.
 *
 * Samples @p tournament_size candidates at random and returns the one with
 * the lowest fitness (minimisation). Uses a thread-local RNG so it is safe
 * to call concurrently from an OpenMP parallel region.
 *
 * @tparam GeneType The gene type of the population.
 */
template <typename GeneType>
class TournamentSelection : public Selection<GeneType> {
public:
    /**
     * @brief Constructs the selector.
     * @param tournament_size Number of candidates per tournament. Must be >= 1.
     * @throws std::invalid_argument if tournament_size is 0.
     */
    explicit TournamentSelection(std::size_t tournament_size);

    /**
     * @brief Runs one tournament and returns the winner.
     * @param population Source population. Must not be empty.
     * @return Const reference to the selected individual.
     */
    const Individual<GeneType>& select(const Population<GeneType>& population) override;
};
```

---

## Common Mistakes

| Mistake | Why it's bad |
|---|---|
| `@brief Returns the fitness.` on `getFitness()` | Repeats the obvious — describe the *contract*, not the name |
| Documenting `private` members | Clutters Doxygen output with implementation noise |
| Missing `@tparam` on template classes | Doxygen won't link template parameters in the HTML |
| Using `//` single-line comments for class-level docs | Doxygen ignores `//` — use `/** */` or `/*! */` |

---

## Namespace Reference

| Namespace | Used for |
|---|---|
| `galib` | Core types: `Algorithm`, `Population`, `Individual`, all operators |
| `galib::benchmark` | Built-in test functions (Rastrigin, Sphere, …) |
| `galib::utils` | Builders, loggers, factories |
| `galib::cuda` | CUDA-accelerated variants |
| `galib::internal` | MPI serialisers / communicators — internal, do not document publicly |
