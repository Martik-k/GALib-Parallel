#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H

#pragma once

#include "core/Individual.h"
#include "algorithms/island/communication/buffers/MigrationBuffer.h"
#include <vector>
#include <cstddef>

namespace galib {

    template <typename GeneType>
    class Communicator {
    public:
        virtual ~Communicator() = default;

        virtual void startReceiving(MigrationBuffer<GeneType>& target_buffer) = 0;

        virtual void update() = 0;

        virtual void stopReceiving() = 0;

        virtual void send(const std::vector<Individual<GeneType>>& deme, std::size_t destination_rank) = 0;

        virtual void broadcast(const std::vector<Individual<GeneType>>& deme,
                               const std::vector<std::size_t>& destination_ranks) = 0;

        virtual Individual<GeneType> allReduceBest(Individual<GeneType> individual) const = 0;

        [[nodiscard]] virtual std::size_t getRank() const = 0;

        [[nodiscard]] virtual std::size_t getSize() const = 0;
    };
}

#endif // COMMUNICATOR_H
