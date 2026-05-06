#ifndef ONE_WAY_RING_TOPOLOGY_H
#define ONE_WAY_RING_TOPOLOGY_H

#pragma once

#include "algorithms/island/topology/Topology.h"

namespace galib {

    /**
     * @brief A topology where islands are connected in a unidirectional circle.
     * 
     * Low connectivity that helps preserve population diversity by slowing
     * down the spread of elite individuals.
     */
    class OneWayRingTopology : public Topology {
    public:
        /** @brief Constructs a unidirectional ring topology for @p num_nodes nodes. */
        explicit OneWayRingTopology(const std::size_t num_nodes) : Topology(num_nodes) {}

        /**
         * @brief Returns the next node as output and previous node as input.
         * @param node Local node ID.
         * @return Neighbors in a one-way ring.
         */
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