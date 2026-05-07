#ifndef MIGRATION_BUFFER_H
#define MIGRATION_BUFFER_H

#pragma once

#include "core/Individual.h"
#include <vector>

namespace galib::internal {
    /**
     * @brief Interface for a thread-safe buffer that stores incoming migrants.
     * 
     * Implementations must handle asynchronous arrivals from the network
     * and allow the GA loop to retrieve them in batches.
     * 
     * @tparam GeneType The gene type of the individuals.
     */
    template <typename GeneType>
    class MigrationBuffer {
    public:
        virtual ~MigrationBuffer() = default;

        /**
         * @brief Adds a batch of individuals to the buffer.
         * @param incoming_deme_batch The individuals that arrived from another island.
         */
        virtual void push(std::vector<Individual<GeneType>>&& incoming_deme_batch) = 0;

        /**
         * @brief Retrieves and removes all currently stored migrants.
         * @param target Vector where the migrants will be moved to.
         */
        virtual void popAll(std::vector<Individual<GeneType>>& target) = 0;

        /**
         * @brief Checks if there are any migrants waiting in the buffer.
         * @return True if migrants are available, false otherwise.
         */
        [[nodiscard]] virtual bool hasMigrants() const = 0;
    };
}

#endif // MIGRATION_BUFFER_H