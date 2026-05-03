#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#pragma once

#include <stdexcept>
#include <vector>

namespace galib {

    namespace internal {
        struct NodeLinks {
            std::vector<std::size_t> neighbors_out;
            std::vector<std::size_t> neighbors_in;
        };
    }

    class Topology {
    private:
        std::size_t num_nodes_m;

    protected:
        [[nodiscard]] std::size_t getNumNodes() const { return num_nodes_m; }

    public:
        virtual ~Topology() = default;

        explicit Topology(const std::size_t num_nodes) : num_nodes_m(num_nodes) {
            if (num_nodes < 2) {
                throw std::invalid_argument("Number of nodes must be greater than 1");
            }
        };

        [[nodiscard]] virtual internal::NodeLinks getLinks(std::size_t node) const = 0;
    };

}

#endif // TOPOLOGY_H