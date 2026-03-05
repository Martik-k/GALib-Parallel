#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include "Config.h"
#include <yaml-cpp/yaml.h>
#include <string>
#include <stdexcept>

namespace galib {
    namespace utils {

        class ConfigParser {
        public:
            static Config parse(const std::string& filepath) {
                Config config;
                try {
                    YAML::Node node = YAML::LoadFile(filepath);

                    config.problem.name = node["problem"]["name"].as<std::string>();
                    config.problem.dimensions = node["problem"]["dimensions"].as<std::size_t>();
                    config.problem.lower_bound = node["problem"]["lower_bound"].as<double>();
                    config.problem.upper_bound = node["problem"]["upper_bound"].as<double>();

                    config.algorithm.pop_size = node["algorithm"]["pop_size"].as<std::size_t>();
                    config.algorithm.max_generations = node["algorithm"]["max_generations"].as<std::size_t>();
                    config.algorithm.mutation_rate = node["algorithm"]["mutation_rate"].as<double>();
                    config.algorithm.crossover_rate = node["algorithm"]["crossover_rate"].as<double>();

                    if (node["algorithm"]["selection"]) {
                        config.algorithm.selection.type =
                            node["algorithm"]["selection"]["type"].as<std::string>("Tournament");
                        config.algorithm.selection.tournament_size =
                            node["algorithm"]["selection"]["tournament_size"].as<int>();
                    } else {
                        config.algorithm.selection.type = "Tournament";
                        config.algorithm.selection.tournament_size = 3;
                    }

                    config.output.log_file = node["output"]["log_file"].as<std::string>("evolution_history.csv");

                } catch (const YAML::Exception& e) {
                    throw std::runtime_error("Error parsing YAML config: " + std::string(e.what()));
                }

                return config;
            }
        };

    }
}

#endif // CONFIG_PARSER_H