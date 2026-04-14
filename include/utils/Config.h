#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <cstddef>

namespace galib {
    namespace utils {

        struct Config {
            struct Problem {
                std::string name;
                std::size_t dimensions;
                double lower_bound;
                double upper_bound;
            } problem;

            struct Algorithm {
                std::string type;
                std::size_t pop_size;
                std::size_t max_generations;
                double mutation_rate;
                double crossover_rate;
                std::string backend;

                struct Selection {
                    std::string type;
                    std::string scope;
                    int tournament_size;
                } selection;

                struct Cellular {
                std::size_t rows;
                    std::size_t cols;
                    bool use_local_elitism;
                } cellular;
            } algorithm;

            struct Output {
                std::string log_file;
            } output;
        };

    }
}

#endif // CONFIG_H