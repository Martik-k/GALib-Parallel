#pragma once

#include "utils/Config.h"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <string>

namespace galib::utils {

class ConfigParser {
public:
    static Config parse(const std::string& path) {
        YAML::Node root;
        try {
            root = YAML::LoadFile(path);
        } catch (const YAML::Exception& e) {
            throw std::runtime_error("Failed to load config file '" + path + "': " + e.what());
        }

        Config cfg;

        if (const auto& p = root["problem"]) {
            cfg.problem.name        = p["name"].as<std::string>(cfg.problem.name);
            cfg.problem.dimensions  = p["dimensions"].as<std::size_t>(cfg.problem.dimensions);
            cfg.problem.lower_bound = p["lower_bound"].as<double>(cfg.problem.lower_bound);
            cfg.problem.upper_bound = p["upper_bound"].as<double>(cfg.problem.upper_bound);
        }

        if (const auto& a = root["algorithm"]) {
            cfg.algorithm.pop_size        = a["pop_size"].as<std::size_t>(cfg.algorithm.pop_size);
            cfg.algorithm.max_generations = a["max_generations"].as<std::size_t>(cfg.algorithm.max_generations);
            cfg.algorithm.mutation_rate   = a["mutation_rate"].as<double>(cfg.algorithm.mutation_rate);
            cfg.algorithm.crossover_rate  = a["crossover_rate"].as<double>(cfg.algorithm.crossover_rate);
            cfg.algorithm.use_elitism     = a["use_elitism"].as<bool>(cfg.algorithm.use_elitism);
            cfg.algorithm.backend         = a["backend"].as<std::string>(cfg.algorithm.backend);

            if (const auto& s = a["selection"]) {
                cfg.algorithm.selection.type            = s["type"].as<std::string>(cfg.algorithm.selection.type);
                cfg.algorithm.selection.tournament_size = s["tournament_size"].as<int>(cfg.algorithm.selection.tournament_size);
            }
        }

        if (const auto& o = root["output"]) {
            cfg.output.log_file = o["log_file"].as<std::string>("");
        }

        return cfg;
    }
};

} // namespace galib::utils
