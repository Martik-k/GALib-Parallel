#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#pragma once

#include <stdexcept>
#include <vector>

namespace galib {

    namespace internal {
        /**
         * @brief Represents the incoming and outgoing links for a specific node in a graph.
         */
        struct NodeLinks {
            /** @brief IDs of nodes that this node sends migrants to. */
            std::vector<std::size_t> neighbors_out;
            /** @brief IDs of nodes that this node receives migrants from. */
            std::vector<std::size_t> neighbors_in;
        };
    }

    /**
     * @brief Abstract base class for archipelago topologies.
     * 
     * Defines the logical connectivity between islands in a distributed parallel GA.
     */
    class Topology {
    private:
        std::size_t num_nodes_m;

    protected:
        /** @brief Returns the total number of nodes in the topology. */
        [[nodiscard]] std::size_t getNumNodes() const { return num_nodes_m; }

    public:
        virtual ~Topology() = default;

        /**
         * @brief Constructs a topology for a specific number of nodes.
         * @param num_nodes Total islands in the archipelago. Must be >= 2.
         * @throws std::invalid_argument if num_nodes < 2.
         */
        explicit Topology(const std::size_t num_nodes) : num_nodes_m(num_nodes) {
            if (num_nodes < 2) {
                throw std::invalid_argument("Number of nodes must be greater than 1");
            }
        };

        /**
         * @brief Determines the neighbors for a specific island rank.
         * @param node The rank/ID of the local island.
         * @return Structure containing input and output neighbor lists.
         */
        [[nodiscard]] virtual internal::NodeLinks getLinks(std::size_t node) const = 0;
    };

}

#endif // TOPOLOGY_H