#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "algorithms/standard/StandardGA.h"
#include "benchmarks/RastriginFunction.h"
#include "benchmarks/SphereFunction.h"
#include "core/FitnessFunction.h"
#include "core/Population.h"

// Custom function with NO GPU kernel — forces CPU-callback path on StandardGACUDA.
// Proves the template works for any user-defined FitnessFunction.
template <typename GeneType = double>
class AckleyFunction : public galib::FitnessFunction<GeneType> {
public:
    explicit AckleyFunction(std::size_t dims)
        : dims_m(dims) {}

    double evaluate(const std::vector<GeneType>& x) const override {
        constexpr double a = 20.0, b = 0.2, c = 2.0 * 3.14159265358979323846;
        double sum_sq = 0, sum_cos = 0;
        for (auto v : x) {
            sum_sq  += static_cast<double>(v) * static_cast<double>(v);
            sum_cos += std::cos(c * static_cast<double>(v));
        }
        const double n = static_cast<double>(x.size());
        return -a * std::exp(-b * std::sqrt(sum_sq / n))
               - std::exp(sum_cos / n)
               + a + std::exp(1.0);
    }

    std::size_t size()                          const override { return dims_m; }
    // No name() override → empty string → CPU callback path on GPU
    GeneType    getLowerBound(std::size_t)      const override { return GeneType(-32.768); }
    GeneType    getUpperBound(std::size_t)      const override { return GeneType( 32.768); }

private:
    std::size_t dims_m;
};
#include "operators/crossover/SinglePointCrossover.h"
#include "operators/mutation/GaussianMutation.h"
#include "operators/selection/TournamentSelection.h"

#ifdef GALIB_WITH_CUDA
#include "algorithms/standard/StandardGACUDA.h"
#endif

using namespace galib;
using ms = std::chrono::duration<double, std::milli>;

// ── helpers ──────────────────────────────────────────────────────────────────

struct Result {
    double wall_ms;
    double best_fitness;
};

static auto makeOps() {
    struct Ops {
        std::unique_ptr<Selection<double>>  sel;
        std::unique_ptr<Mutation<double>>   mut;
        std::unique_ptr<Crossover<double>>  cross;
    };
    return Ops{
        std::make_unique<TournamentSelection<double>>(3),
        std::make_unique<GaussianMutation<double>>(0.1),
        std::make_unique<SinglePointCrossover<double>>()
    };
}

// Redirect cout to /dev/null during the timed run to suppress progress output
struct SilentRun {
    SilentRun()  { old = std::cout.rdbuf(null.rdbuf()); }
    ~SilentRun() { std::cout.rdbuf(old); }
private:
    std::ofstream null{ "/dev/null" };
    std::streambuf* old;
};

static Result runCPU(FitnessFunction<double>& ff,
                     int pop_size, int dims, int generations) {
    auto ops = makeOps();
    StandardGA<double> ga(ff, std::move(ops.sel), std::move(ops.mut), std::move(ops.cross),
                          0.05, 0.8, generations, true);

    Population<double> pop(pop_size, dims);
    pop.initialize(ff.getLowerBound(0), ff.getUpperBound(0));

    SilentRun _;
    auto t0 = std::chrono::steady_clock::now();
    ga.run(pop);
    auto t1 = std::chrono::steady_clock::now();

    return { ms(t1 - t0).count(), pop.getBestIndividual().getFitness() };
}

#ifdef GALIB_WITH_CUDA
static Result runGPU(FitnessFunction<double>& ff,
                     int pop_size, int dims, int generations) {
    auto ops = makeOps();
    cuda::StandardGACUDA<double> ga(ff, std::move(ops.sel), std::move(ops.mut), std::move(ops.cross),
                                    0.05, 0.8, generations, true);

    Population<double> pop(pop_size, dims);
    pop.initialize(ff.getLowerBound(0), ff.getUpperBound(0));

    SilentRun _;
    auto t0 = std::chrono::steady_clock::now();
    ga.run(pop);
    auto t1 = std::chrono::steady_clock::now();

    return { ms(t1 - t0).count(), pop.getBestIndividual().getFitness() };
}
#endif

static std::ofstream g_csv;

static void csvRow(const std::string& label, const std::string& eval_type,
                   int pop, int dims, const Result& cpu, const Result* gpu) {
    if (!g_csv.is_open()) return;
    double speedup = gpu ? cpu.wall_ms / gpu->wall_ms : 1.0;
    double gpu_ms  = gpu ? gpu->wall_ms  : 0.0;
    double gpu_fit = gpu ? gpu->best_fitness : 0.0;
    g_csv << label << "," << eval_type << "," << pop << "," << dims << ","
          << std::fixed << std::setprecision(3)
          << cpu.wall_ms << "," << cpu.best_fitness << ","
          << gpu_ms << "," << gpu_fit << ","
          << speedup << "\n";
}

static void printRow(const std::string& label, int pop, int dims,
                     const Result& cpu, const Result* gpu = nullptr,
                     const std::string& eval_type = "cpu_only") {
    std::cout << std::left  << std::setw(10) << label
              << std::right << std::setw(7)  << pop
              << std::setw(6)  << dims
              << std::setw(10) << std::fixed << std::setprecision(1) << cpu.wall_ms
              << std::setw(12) << std::fixed << std::setprecision(3) << cpu.best_fitness;
#ifdef GALIB_WITH_CUDA
    if (gpu) {
        double speedup = cpu.wall_ms / gpu->wall_ms;
        std::cout << std::setw(10) << std::fixed << std::setprecision(1) << gpu->wall_ms
                  << std::setw(12) << std::fixed << std::setprecision(3) << gpu->best_fitness
                  << std::setw(9)  << std::fixed << std::setprecision(2) << speedup << "x";
    }
#endif
    std::cout << "\n";
    csvRow(label, eval_type, pop, dims, cpu, gpu);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    constexpr int GENERATIONS = 300;

    // Optional CSV output path (e.g. benchmark_ga_cuda --csv results.csv)
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--csv") {
            g_csv.open(argv[i + 1]);
            if (g_csv.is_open())
                g_csv << "function,eval_type,pop_size,dims,cpu_ms,cpu_fitness,"
                         "gpu_ms,gpu_fitness,speedup\n";
        }
    }

    std::cout << "\n=== GALib Benchmark: StandardGA (CPU) vs StandardGACUDA (GPU) ===\n";
    std::cout << "Fitness: Rastrigin | Generations: " << GENERATIONS << "\n\n";

#ifdef GALIB_WITH_CUDA
    std::cout << std::left  << std::setw(10) << "Function"
              << std::right << std::setw(7)  << "PopSz"
              << std::setw(6)  << "Dims"
              << std::setw(10) << "CPU ms"
              << std::setw(12) << "CPU fit"
              << std::setw(10) << "GPU ms"
              << std::setw(12) << "GPU fit"
              << std::setw(9)  << "Speedup"
              << "\n" << std::string(76, '-') << "\n";
#else
    std::cout << std::left  << std::setw(10) << "Function"
              << std::right << std::setw(7)  << "PopSz"
              << std::setw(6)  << "Dims"
              << std::setw(10) << "CPU ms"
              << std::setw(12) << "CPU fit"
              << "\n" << std::string(45, '-') << "\n";
#endif

    // Benchmark matrix
    struct Case { int pop; int dims; };
    const std::vector<Case> cases = {
        { 500,   10 },
        { 500,   50 },
        { 2000,  10 },
        { 2000,  50 },
        { 5000,  50 },
        { 10000, 50 },
    };

    for (auto& c : cases) {
        benchmark::RastriginFunction<double> ff(c.dims, -5.12, 5.12);
        Result cpu = runCPU(ff, c.pop, c.dims, GENERATIONS);
#ifdef GALIB_WITH_CUDA
        Result gpu = runGPU(ff, c.pop, c.dims, GENERATIONS);
        printRow("Rastrigin", c.pop, c.dims, cpu, &gpu, "gpu_kernel");
#else
        printRow("Rastrigin", c.pop, c.dims, cpu, nullptr, "cpu_only");
#endif
    }

    std::cout << std::string(76, '-') << "\n\n";

    // Sphere — GPU kernel path vs CPU
    std::cout << "=== Sphere (GPU kernel evaluation enabled) ===\n";
#ifdef GALIB_WITH_CUDA
    std::cout << std::left  << std::setw(10) << "Function"
              << std::right << std::setw(7)  << "PopSz"
              << std::setw(6)  << "Dims"
              << std::setw(10) << "CPU ms"
              << std::setw(12) << "CPU fit"
              << std::setw(10) << "GPU ms"
              << std::setw(12) << "GPU fit"
              << std::setw(9)  << "Speedup"
              << "\n" << std::string(76, '-') << "\n";
    for (auto& c : cases) {
        benchmark::SphereFunction<double> ff(c.dims, -5.12, 5.12);
        Result cpu = runCPU(ff, c.pop, c.dims, GENERATIONS);
        Result gpu = runGPU(ff, c.pop, c.dims, GENERATIONS);
        printRow("Sphere", c.pop, c.dims, cpu, &gpu, "gpu_kernel");
    }
    std::cout << std::string(76, '-') << "\n";
    std::cout << "\nNote: Rastrigin uses CPU callback (virtual evaluate). "
                 "Sphere uses GPU kernel (name()=\"Sphere\").\n";
#endif

    // Custom function — no GPU kernel, uses CPU callback on StandardGACUDA
    std::cout << "\n=== Ackley (custom function — CPU callback path on GPU) ===\n";
#ifdef GALIB_WITH_CUDA
    std::cout << std::left  << std::setw(10) << "Function"
              << std::right << std::setw(7)  << "PopSz"
              << std::setw(6)  << "Dims"
              << std::setw(10) << "CPU ms"
              << std::setw(12) << "CPU fit"
              << std::setw(10) << "GPU ms"
              << std::setw(12) << "GPU fit"
              << std::setw(9)  << "Speedup"
              << "\n" << std::string(76, '-') << "\n";
    for (auto& c : cases) {
        AckleyFunction<double> ff(c.dims);
        Result cpu = runCPU(ff, c.pop, c.dims, GENERATIONS);
        Result gpu = runGPU(ff, c.pop, c.dims, GENERATIONS);
        printRow("Ackley", c.pop, c.dims, cpu, &gpu, "cpu_callback");
    }
    std::cout << std::string(76, '-') << "\n";
    std::cout << "\nAckley has no GPU kernel: StandardGACUDA uses the CPU callback for evaluation\n"
                 "but still runs selection/crossover/mutation on the GPU.\n";
#else
    for (auto& c : cases) {
        AckleyFunction<double> ff(c.dims);
        Result cpu = runCPU(ff, c.pop, c.dims, GENERATIONS);
        printRow("Ackley", c.pop, c.dims, cpu, nullptr, "cpu_only");
    }
#endif

    return 0;
}
