#ifndef FULLY_CONNECTED_TOPOLOGY_H
#define FULLY_CONNECTED_TOPOLOGY_H

#pragma once

#include "algorithms/island/topology/Topology.h"

namespace galib {

    /**
     * @brief A topology where every island is connected to every other island.
     * 
     * High connectivity results in rapid convergence as elite individuals
     * spread throughout the archipelago quickly.
     */
    class FullyConnectedTopology : public Topology {
    public:
        /** @brief Constructs a fully connected topology for @p num_nodes nodes. */
        explicit FullyConnectedTopology(const std::size_t num_nodes) : Topology(num_nodes) {}

        /**
         * @brief Returns all nodes (except @p node) as both input and output neighbors.
         * @param node Local node ID.
         * @return Links to every other node in the network.
         */
        internal::NodeLinks getLinks(const std::size_t node) const override {
            internal::NodeLinks links;
            const std::size_t num_nodes = getNumNodes();
            links.neighbors_out.reserve(num_nodes - 1);
            links.neighbors_in.reserve(num_nodes - 1);

            for (std::size_t i = 0; i < num_nodes; ++i) {
                if (i != node) {
                    links.neighbors_out.push_back(i);
                    links.neighbors_in.push_back(i);
                }
            }

            return links;
        }

    };

}

#endif // FULLY_CONNECTED_TOPOLOGY_H