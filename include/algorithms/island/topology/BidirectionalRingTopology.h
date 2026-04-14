#ifndef BIDIRECTIONAL_RING_TOPOLOGY_H
#define BIDIRECTIONAL_RING_TOPOLOGY_H

#pragma once

#include "algorithms/island/topology/Topology.h"

namespace galib {

    class BidirectionalRingTopology : public Topology {
    public:
        explicit BidirectionalRingTopology(const std::size_t num_nodes) : Topology(num_nodes) {}

        NodeLinks getLinks(const std::size_t node) const override {
            const std::size_t num_nodes = getNumNodes();
            return {
                .neighbors_out = {(node - 1 + num_nodes) % num_nodes, (node + 1) % num_nodes},
                .neighbors_in = {(node - 1 + num_nodes) % num_nodes, (node + 1) % num_nodes}
            };
        }
    };

}

#endif // BIDIRECTIONAL_RING_TOPOLOGY_H