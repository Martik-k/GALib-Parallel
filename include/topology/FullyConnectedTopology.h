#ifndef FULLY_CONNECTED_TOPOLOGY_H
#define FULLY_CONNECTED_TOPOLOGY_H

#pragma once

#include "topology/Topology.h"

namespace galib {

    class FullyConnectedTopology : public Topology {
    public:
        explicit FullyConnectedTopology(const std::size_t num_nodes) : Topology(num_nodes) {}

        NodeLinks getLinks(const std::size_t node) const override {
            NodeLinks links;
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