#pragma once

#include <cstddef>
#include <string>

namespace galib::utils {

struct SelectionConfig {
    std::string type           = "tournament";
    int         tournament_size = 3;
};

struct AlgorithmConfig {
    std::size_t    pop_size        = 100;
    std::size_t    max_generations = 1000;
    double         mutation_rate   = 0.05;
    double         crossover_rate  = 0.8;
    bool           use_elitism     = true;
    std::string    backend         = "standard";
    SelectionConfig selection;
};

struct ProblemConfig {
    std::string name        = "Sphere";
    std::size_t dimensions  = 2;
    double      lower_bound = -5.12;
    double      upper_bound =  5.12;
};

struct OutputConfig {
    std::string log_file;
};

struct Config {
    AlgorithmConfig algorithm;
    ProblemConfig   problem;
    OutputConfig    output;
};

} // namespace galib::utils
