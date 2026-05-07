# Configuration Reference {#config_reference}

GALib-Parallel is driven by a single **YAML configuration file** that selects the
algorithm, tunes its parameters, wires up genetic operators, and controls logging.
You never write boilerplate construction code — just point `AlgorithmBuilder` at your
file and call `build()`.

```cpp
#include "utils/AlgorithmBuilder.h"

auto algo = galib::utils::AlgorithmBuilder<double>::build("my_config.yaml", fitness_fn);
algo->run(population);
```

---

## File Layout

A config file has two top-level keys:

| Key | Required | Description |
|-----|----------|-------------|
| `algorithm` | **Yes** | Selects and parameterises the algorithm |
| `output` | No | Console / file logging (defaults to off) |

---

## `algorithm` — Common Parameters

These keys live directly under `algorithm:` and apply to **every** algorithm type.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"standard"` | Which algorithm to run. See [Algorithm Types](#config_types). |
| `max_generations` | integer | `100` | Stop after this many generations. |
| `mutation_rate` | float | `0.05` | Per-individual probability of mutation `[0.0, 1.0]`. |
| `crossover_rate` | float | `0.8` | Probability that two parents exchange genetic material `[0.0, 1.0]`. |
| `use_elitism` | bool | `true` | Carry the best individual unchanged into the next generation. |
| `threads` | integer | `1` | OpenMP thread count. `0` uses all available hardware threads. |

---

## Algorithm Types {#config_types}

Set `algorithm.type` to one of the following strings:

| Value | Parallelism | Requires |
|-------|-------------|---------|
| `"standard"` | OpenMP / optional CUDA | — |
| `"cellular"` | OpenMP | — |
| `"differential_evolution"` | OpenMP | — |
| `"island"` | MPI + OpenMP | library built with `-DGALIB_HAS_MPI=ON` |

---

### Standard GA (`"standard"`)

A classic generational genetic algorithm with optional GPU-accelerated fitness evaluation.

Extra key under `algorithm.standard`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_cuda` | bool | `false` | Offload fitness evaluation to CUDA. Requires build with `-DGALIB_WITH_CUDA=ON`; falls back to CPU with a warning if unavailable. |

**Minimal example:**

```yaml
algorithm:
  type: "standard"
  max_generations: 200
  mutation_rate: 0.05
  crossover_rate: 0.8
  threads: 4

  selection:
    type: "tournament"
    tournament_size: 3
  mutation:
    type: "gaussian"
    sigma: 0.1
  crossover:
    type: "single_point"
```

**CUDA example:**

```yaml
algorithm:
  type: "standard"
  max_generations: 500
  mutation_rate: 0.05
  crossover_rate: 0.8

  standard:
    use_cuda: true

  selection:
    type: "tournament"
    tournament_size: 5
  mutation:
    type: "gaussian"
    sigma: 0.05
  crossover:
    type: "arithmetic"
```

---

### Cellular GA (`"cellular"`)

Individuals are arranged on a 2-D grid. Selection and mating are restricted to
Moore-neighbourhood cells, which promotes genetic diversity and spatial structure.

Extra keys under `algorithm.cellular`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rows` | integer | `10` | Number of rows in the spatial grid. |
| `cols` | integer | `10` | Number of columns in the spatial grid. |
| `use_local_elitism` | bool | `true` | Replace a cell only if the offspring is better than the current occupant. |

> **Note:** `selection.type` must be `"best_neighbor"` or `"tournament"` for Cellular GA —
> both map to the neighbourhood-best selection strategy.

**Example:**

```yaml
algorithm:
  type: "cellular"
  max_generations: 300
  mutation_rate: 0.08
  crossover_rate: 0.7
  threads: 8

  cellular:
    rows: 20
    cols: 20
    use_local_elitism: true

  selection:
    type: "best_neighbor"
  mutation:
    type: "gaussian"
    sigma: 0.15
  crossover:
    type: "uniform"
```

---

### Differential Evolution (`"differential_evolution"`)

A population-based optimiser that constructs trial vectors by combining three
randomly chosen individuals. Genetic operator keys (`selection`, `crossover`) are
**ignored** — DE has its own fixed recombination scheme.

Extra keys under `algorithm.differential_evolution`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `f_weight` | float | `0.8` | Differential weight (scale factor) `[0.0, 2.0]`. Controls how aggressively difference vectors are scaled. |

> `crossover_rate` (top-level) acts as the **CR** parameter in the DE/rand/1/bin scheme.

**Example:**

```yaml
algorithm:
  type: "differential_evolution"
  max_generations: 1000
  crossover_rate: 0.9
  threads: 4

  differential_evolution:
    f_weight: 0.5

  mutation:
    type: "gaussian"
    sigma: 0.1
```

---

### Island GA (`"island"`)

Multiple independent sub-populations (islands) evolve in parallel MPI processes
and periodically exchange individuals (migrate) according to a chosen topology.

Requires the library to be built with MPI support (`-DGALIB_HAS_MPI=ON`).
Pass the MPI communicator to `AlgorithmBuilder::build()`:

```cpp
auto algo = galib::utils::AlgorithmBuilder<double>::build(
    "island_config.yaml", fitness_fn, MPI_COMM_WORLD);
```

Extra keys under `algorithm.island`:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `topology` | string | `"fully_connected"` | How islands are connected. See table below. |
| `migration_interval` | integer | `50` | Migrate every N generations. |
| `migration_size` | integer | `5` | Number of individuals sent per migration event. |
| `immigration_quota` | float | `0.1` | Fraction of the local population that can be replaced by immigrants `[0.0, 1.0]`. |
| `buffer_capacity` | integer | `10` | Size of each island's incoming migration buffer (circular). |

**Topology options:**

| Value | Description |
|-------|-------------|
| `"fully_connected"` | Every island communicates with every other island. |
| `"one_way_ring"` | Each island sends migrants to the next island in a ring. |
| `"bidirectional_ring"` | Each island exchanges migrants with both its left and right neighbours. |

**Migration selector** — which individuals leave an island (`algorithm.island.selector`):

| `type` | Description |
|--------|-------------|
| `"elitism"` | *(default)* Send the best individuals. |

**Migration replacer** — how immigrants enter an island (`algorithm.island.replacer`):

| `type` | Description |
|--------|-------------|
| `"worst"` | *(default)* Immigrants replace the worst locals. |

**Example:**

```yaml
algorithm:
  type: "island"
  max_generations: 500
  mutation_rate: 0.08
  crossover_rate: 0.85
  use_elitism: true
  threads: 2

  island:
    topology: "bidirectional_ring"
    migration_interval: 20
    migration_size: 5
    immigration_quota: 0.1
    buffer_capacity: 30

    selector:
      type: "elitism"
    replacer:
      type: "worst"

  selection:
    type: "tournament"
    tournament_size: 4
  mutation:
    type: "uniform"
  crossover:
    type: "arithmetic"
```

---

## Genetic Operators

### Selection (`algorithm.selection`)

Chooses parents for recombination each generation.

| `type` | Extra keys | Description |
|--------|-----------|-------------|
| `"tournament"` | `tournament_size` (int, default `3`) | Randomly sample N individuals and pick the best. Larger values increase selection pressure. |
| `"best_neighbor"` | — | Cellular GA only: pick the best individual in the Moore neighbourhood. |

### Mutation (`algorithm.mutation`)

Randomly perturbs genes after crossover.

| `type` | Extra keys | Description |
|--------|-----------|-------------|
| `"gaussian"` | `sigma` (float, default `0.1`) | Add Gaussian noise N(0, σ) to each gene. Good general-purpose choice. |
| `"uniform"` | — | Replace each gene with a uniform random value in `[lower_bound, upper_bound]`. |
| `"boundary"` | — | Replace each gene with either the lower or upper bound of the search space. Useful for boundary exploitation. |

### Crossover (`algorithm.crossover`)

Combines genetic material from two parent individuals.

| `type` | Description |
|--------|-------------|
| `"single_point"` | Split parents at one random locus and swap tails. |
| `"arithmetic"` | Produce offspring as a weighted average of the two parents. |
| `"uniform"` | Each gene is independently taken from one parent with equal probability. |

---

## `output` — Logging

Both loggers are disabled by default.

```yaml
output:
  console:
    enabled: true      # Print progress to stdout
    interval: 10       # Print every N generations (default: 1)
  file:
    enabled: false     # Write CSV to disk
    interval: 1        # Log every N generations
    path: "logs/evolution.csv"  # Output path
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `console.enabled` | bool | `false` | Enable stdout progress output. |
| `console.interval` | integer | `1` | Log every N-th generation to the console. |
| `file.enabled` | bool | `false` | Enable CSV file logging. |
| `file.interval` | integer | `1` | Write every N-th generation to the file. |
| `file.path` | string | `"evolution.csv"` | Destination path (directories must exist). |

---

## Default Values at a Glance

| Parameter | Default |
|-----------|---------|
| `max_generations` | `100` |
| `mutation_rate` | `0.05` |
| `crossover_rate` | `0.8` |
| `use_elitism` | `true` |
| `threads` | `1` (0 = all cores) |
| `standard.use_cuda` | `false` |
| `cellular.rows` / `cols` | `10` |
| `cellular.use_local_elitism` | `true` |
| `differential_evolution.f_weight` | `0.8` |
| `island.topology` | `"fully_connected"` |
| `island.migration_interval` | `50` |
| `island.migration_size` | `5` |
| `island.immigration_quota` | `0.1` |
| `island.buffer_capacity` | `10` |
| `selection.tournament_size` | `3` |
| `mutation.sigma` (gaussian) | `0.1` |

---

## Complete Example

See [configs/full_config_example.yaml](../../configs/full_config_example.yaml) for a single
file that lists every supported parameter with inline comments.

@see galib::utils::AlgorithmBuilder
@see galib::AlgorithmConfig
