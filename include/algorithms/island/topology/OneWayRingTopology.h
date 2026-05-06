#ifndef ONE_WAY_RING_TOPOLOGY_H
#define ONE_WAY_RING_TOPOLOGY_H

#pragma once

#include "algorithms/island/topology/Topology.h"

namespace galib {

    class OneWayRingTopology : public Topology {
    public:
        explicit OneWayRingTopology(const std::size_t num_nodes) : Topology(num_nodes) {}

        [[nodiscard]] internal::NodeLinks getLinks(const std::size_t node) const override {
            const std::size_t num_nodes = getNumNodes();
            return {
                .neighbors_out = {(node + 1) % num_nodes},
                .neighbors_in = {(node - 1 + num_nodes) % num_nodes}
            };
        }
    };

}

#endif // ONE_WAY_RING_TOPOLOGY_H