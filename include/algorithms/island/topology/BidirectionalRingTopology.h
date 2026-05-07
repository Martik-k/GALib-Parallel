#ifndef BIDIRECTIONAL_RING_TOPOLOGY_H
#define BIDIRECTIONAL_RING_TOPOLOGY_H

#pragma once

#include "algorithms/island/topology/Topology.h"

namespace galib {

    /**
     * @brief A topology where islands are connected in a bidirectional circle.
     * 
     * Similar to the one-way ring but allows genetic material to flow
     * in both directions between neighbors.
     */
    class BidirectionalRingTopology : public Topology {
    public:
        /** @brief Constructs a bidirectional ring topology for @p num_nodes nodes. */
        explicit BidirectionalRingTopology(const std::size_t num_nodes) : Topology(num_nodes) {}

        /**
         * @brief Returns both adjacent nodes as input and output neighbors.
         * @param node Local node ID.
         * @return Neighbors in a two-way ring.
         */
        internal::NodeLinks getLinks(const std::size_t node) const override {
            const std::size_t num_nodes = getNumNodes();
            return {
                .neighbors_out = {(node - 1 + num_nodes) % num_nodes, (node + 1) % num_nodes},
                .neighbors_in = {(node - 1 + num_nodes) % num_nodes, (node + 1) % num_nodes}
            };
        }
    };

}

#endif // BIDIRECTIONAL_RING_TOPOLOGY_H