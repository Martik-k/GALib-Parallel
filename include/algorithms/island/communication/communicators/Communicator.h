#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#pragma once

#include "core/Individual.h"
#include "algorithms/island/communication/buffers/MigrationBuffer.h"
#include <vector>
#include <cstddef>

namespace galib::internal {

    /**
     * @brief Interface for network communication between islands.
     * 
     * Handles the low-level details of moving individuals between distributed
     * nodes and synchronizing global state.
     * 
     * @tparam GeneType The gene type of the individuals.
     */
    template <typename GeneType>
    class Communicator {
    public:
        virtual ~Communicator() = default;

        /**
         * @brief Begins asynchronous background receiving of migrants.
         * @param target_buffer The buffer where arrived individuals will be stored.
         */
        virtual void startReceiving(MigrationBuffer<GeneType>& target_buffer) = 0;

        /**
         * @brief Processes any pending network events (e.g. polls for incoming messages).
         */
        virtual void update() = 0;

        /**
         * @brief Stops background communication activities.
         */
        virtual void stopReceiving() = 0;

        /**
         * @brief Sends a deme to a specific destination island.
         * @param deme             The individuals to migrate.
         * @param destination_rank The network ID/rank of the target island.
         */
        virtual void send(const std::vector<Individual<GeneType>>& deme, std::size_t destination_rank) = 0;

        /**
         * @brief Sends a deme to multiple destination islands.
         * @param deme              The individuals to migrate.
         * @param destination_ranks List of target island IDs.
         */
        virtual void broadcast(const std::vector<Individual<GeneType>>& deme,
                               const std::vector<std::size_t>& destination_ranks) = 0;

        /**
         * @brief Global reduction to find the best individual across all islands.
         * 
         * @param individual The local best individual.
         * @return A copy of the absolute best individual found in the entire archipelago.
         */
        virtual Individual<GeneType> allReduceBest(Individual<GeneType> individual) const = 0;

        /**
         * @brief Gets the network rank of the local node.
         * @return The ID of the local island.
         */
        [[nodiscard]] virtual std::size_t getRank() const = 0;

        /**
         * @brief Gets the total number of islands in the network.
         * @return The size of the archipelago.
         */
        [[nodiscard]] virtual std::size_t getSize() const = 0;
    };
}

#endif // COMMUNICATOR_H
