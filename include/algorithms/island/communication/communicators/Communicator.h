#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#pragma once

#include "core/Individual.h"
#include "buffers/MigrationBuffer.h"
#include <vector>
#include <cstddef>

namespace galib {

    /**
     * @brief Abstract interface for asynchronous communication between islands.
     * 
     * The communicator is responsible for the transport of individuals between processes.
     * It handles serialization and background receiving.
     */
    template <typename GeneType>
    class Communicator {
    public:
        virtual ~Communicator() = default;

        /**
         * @brief Starts the background receiving mechanism.
         * @param target_buffer The buffer where incoming migrants will be deposited.
         */
        virtual void startReceiving(MigrationBuffer<GeneType>& target_buffer) = 0;

        /**
         * @brief Polls the underlying transport for new messages and processes them.
         * Should be called periodically (e.g., once per generation).
         */
        virtual void update() = 0;

        /**
         * @brief Stops all background communication activities.
         */
        virtual void stopReceiving() = 0;

        /**
         * @brief Sends a deme to a specific target island asynchronously.
         * @param deme The individuals to send.
         * @param destination_rank The ID/Rank of the destination island.
         */
        virtual void send(const std::vector<Individual<GeneType>>& deme, std::size_t destination_rank) = 0;

        /**
         * @brief Sends a deme to multiple target islands.
         * @param deme The individuals to send.
         * @param destination_ranks List of destination IDs/Ranks.
         */
        virtual void broadcast(const std::vector<Individual<GeneType>>& deme, 
                               const std::vector<std::size_t>& destination_ranks) = 0;
        
        /**
         * @brief Gets the unique rank/ID of the current process in the cluster.
         */
        [[nodiscard]] virtual std::size_t getRank() const = 0;

        /**
         * @brief Gets the total number of processes/islands in the cluster.
         */
        [[nodiscard]] virtual std::size_t getSize() const = 0;
    };
}

#endif // COMMUNICATOR_H
