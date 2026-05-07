#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <fstream>
#include <getopt.h>
#include <atomic>
#include <map>
#include <filesystem>
#include <algorithm>

#include "utils/AlgorithmBuilder.h"
#include "benchmarks/RastriginFunction.h"
#include "core/Population.h"

using namespace galib;

struct BenchmarkConfig {
    std::string config_path = "configs/full_config_example.yaml";
    std::vector<int> threads_list = {1};
    std::string output_csv = "results/benchmark_results.csv";
    int num_genes = 100;
    int population_size = 500;
    int num_runs = 5;  // Number of runs per thread count
    std::string benchmark_type = "time";  // time, quality, convergence
    std::vector<std::string> algorithms = {"standard"};  // standard, de, cellular
};

std::vector<int> parse_threads(const std::string& str) {
    std::vector<int> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stoi(item));
    }
    return result;
}

std::vector<std::string> parse_algorithms(const std::string& str) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(item);
    }
    return result;
}

inline std::chrono::high_resolution_clock::time_point
get_current_time_fenced()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;

    // Parse command line arguments
    static struct option long_options[] = {
        {"config", required_argument, 0, 'c'},
        {"threads", required_argument, 0, 't'},
        {"output", required_argument, 0, 'o'},
        {"num_genes", required_argument, 0, 'g'},
        {"pop_size", required_argument, 0, 'p'},
        {"runs", required_argument, 0, 'r'},
        {"benchmark", required_argument, 0, 'b'},
        {"algorithms", required_argument, 0, 'a'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "c:t:o:g:p:r:b:a:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'c':
                config.config_path = optarg;
                break;
            case 't':
                config.threads_list = parse_threads(optarg);
                break;
            case 'o':
                config.output_csv = optarg;
                break;
            case 'g':
                config.num_genes = std::stoi(optarg);
                break;
            case 'p':
                config.population_size = std::stoi(optarg);
                break;
            case 'r':
                config.num_runs = std::stoi(optarg);
                break;
            case 'b':
                config.benchmark_type = optarg;
                break;
            case 'a':
                config.algorithms = parse_algorithms(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [--config path] [--threads 1,2,4] [--output file.csv] [--num_genes 10] [--pop_size 50] [--runs 5] [--benchmark time|quality|convergence] [--algorithms standard,de,cellular]" << std::endl;
                return 1;
        }
    }

    // Create results directory
    std::filesystem::create_directories("results");

    // Open CSV output
    std::ofstream csv_file(config.output_csv);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Cannot open output file " << config.output_csv << std::endl;
        return 1;
    }
    if (config.benchmark_type == "time") {
        csv_file << "algorithm,threads,time_seconds,speedup,efficiency\n";
    } else if (config.benchmark_type == "quality") {
        csv_file << "algorithm,threads,best_fitness\n";
    } else if (config.benchmark_type == "convergence") {
        csv_file << "algorithm,threads,generation,best_fitness\n";
    }

    benchmark::RastriginFunction<double> fitness_fn(config.num_genes, -5.12, 5.12);

    // Map algorithms to their config files
    std::map<std::string, std::string> algo_configs = {
        {"standard", "configs/full_config_example.yaml"},
        {"de", "configs/full_config_example.yaml"},
        {"cellular", "configs/full_config_example.yaml"}
    };

    for (const auto& algo_type : config.algorithms) {
        std::string config_path = algo_configs.count(algo_type) ? algo_configs[algo_type] : config.config_path;
        YAML::Node config_node = YAML::LoadFile(config_path);
        config_node["algorithm"]["type"] = algo_type;

        if (config.benchmark_type == "time") {
            // Time benchmark with warmup and speedup calculation
            // Warmup run
            {
                auto algo = utils::AlgorithmBuilder<double>::build(config_node, fitness_fn, 1);
                Population<double> population(config.population_size, config.num_genes);
                population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));
                algo->run(population);
            }

            // Run for 1 thread to get T1
            double t1_avg = 0.0;
            {
                std::vector<double> times_1;
                for (int run = 0; run < config.num_runs; ++run) {
                    auto algo = utils::AlgorithmBuilder<double>::build(config_node, fitness_fn, 1);
                    Population<double> population(config.population_size, config.num_genes);
                    population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

                    auto start = get_current_time_fenced();
                    algo->run(population);
                    auto end = get_current_time_fenced();
                    std::chrono::duration<double> elapsed = end - start;
                    times_1.push_back(elapsed.count());
                }
                for (double t : times_1) t1_avg += t;
                t1_avg /= config.num_runs;
            }

            // Always add data point for 1 thread with speedup=1
            csv_file << algo_type << "," << 1 << "," << t1_avg << "," << 1.0 << "," << 1.0 << "\n";

            for (int threads : config.threads_list) {
                if (threads == 1) continue;  // Skip if already added

                std::vector<double> times;
                for (int run = 0; run < config.num_runs; ++run) {
                    auto algo = utils::AlgorithmBuilder<double>::build(config_node, fitness_fn, threads);
                    Population<double> population(config.population_size, config.num_genes);
                    population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

                    auto start = get_current_time_fenced();
                    algo->run(population);
                    auto end = get_current_time_fenced();
                    std::chrono::duration<double> elapsed = end - start;
                    times.push_back(elapsed.count());
                }

                double sum = 0.0;
                for (double t : times) sum += t;
                double avg_time = sum / config.num_runs;

                double speedup = (t1_avg > 0) ? t1_avg / avg_time : 0.0;
                double efficiency = speedup / threads;

                csv_file << algo_type << "," << threads << "," << avg_time << "," << speedup << "," << efficiency << "\n";

                std::cout << "Algorithm: " << algo_type << ", Threads: " << threads << ", Average Time: " << avg_time << " seconds, Speedup: " << speedup << ", Efficiency: " << efficiency << " (over " << config.num_runs << " runs)" << std::endl;
            }
        } else if (config.benchmark_type == "quality") {
            // Quality benchmark: only convergence plot on single thread
            int conv_threads = 1;  // Always use 1 thread for convergence illustration
            YAML::Node modified_config = YAML::Clone(config_node);
            modified_config["output"]["file"]["enabled"] = true;
            modified_config["output"]["file"]["interval"] = 1;
            modified_config["output"]["file"]["path"] = "results/quality_convergence_" + algo_type + ".csv";

            auto algo = utils::AlgorithmBuilder<double>::build(modified_config, fitness_fn, conv_threads);
            Population<double> population(config.population_size, config.num_genes);
            population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

            algo->run(population);

            std::cout << "Algorithm: " << algo_type << ", Convergence data for quality plot saved to results/quality_convergence_" << algo_type << ".csv" << std::endl;
        } else if (config.benchmark_type == "convergence") {
            // Convergence benchmark
            for (int threads : config.threads_list) {
                std::vector<double> best_fitnesses;
                for (int run = 0; run < config.num_runs; ++run) {
                    YAML::Node modified_config = YAML::Clone(config_node);
                    modified_config["output"]["file"]["enabled"] = true;
                    modified_config["output"]["file"]["interval"] = 1;
                    modified_config["output"]["file"]["path"] = "results/convergence_" + algo_type + "_" + std::to_string(threads) + "_run" + std::to_string(run) + ".csv";

                    auto algo = utils::AlgorithmBuilder<double>::build(modified_config, fitness_fn, threads);
                    Population<double> population(config.population_size, config.num_genes);
                    population.initialize(fitness_fn.getLowerBound(0), fitness_fn.getUpperBound(0));

                    algo->run(population);

                    best_fitnesses.push_back(population.getBestIndividual().getFitness());
                }

                // Process convergence logs
                std::map<int, std::vector<double>> gen_to_fitnesses;
                for (int run = 0; run < config.num_runs; ++run) {
                    std::string log_path = "results/convergence_" + algo_type + "_" + std::to_string(threads) + "_run" + std::to_string(run) + ".csv";
                    std::ifstream log_file(log_path);
                    if (log_file.is_open()) {
                        std::string line;
                        std::getline(log_file, line); // header
                        while (std::getline(log_file, line)) {
                            std::stringstream ss(line);
                            std::string gen_str, idx_str, fit_str, geno_str;
                            std::getline(ss, gen_str, ',');
                            std::getline(ss, idx_str, ',');
                            std::getline(ss, fit_str, ',');
                            int gen = std::stoi(gen_str);
                            double fit = std::stod(fit_str);
                            gen_to_fitnesses[gen].push_back(fit);
                        }
                    }
                }

                // Compute average best_fitness per generation
                std::ofstream conv_file("results/convergence.csv", std::ios::app);
                if (!conv_file.is_open()) {
                    conv_file.open("results/convergence.csv", std::ios::out);
                    conv_file << "algorithm,threads,generation,best_fitness\n";
                }
                for (auto& p : gen_to_fitnesses) {
                    int gen = p.first;
                    double sum = 0.0;
                    for (double f : p.second) sum += f;
                    double avg_fit = sum / p.second.size();
                    conv_file << algo_type << "," << threads << "," << gen << "," << avg_fit << "\n";
                }

                std::cout << "Algorithm: " << algo_type << ", Threads: " << threads << ", Convergence data saved to results/convergence.csv" << std::endl;
            }
        }
    }

    csv_file.close();

    return 0;
}