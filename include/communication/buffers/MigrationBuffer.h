#ifndef MIGRATION_BUFFER_H
#define MIGRATION_BUFFER_H

#pragma once

#include "core/Individual.h"
#include <vector>

namespace galib {
    template <typename GeneType>
    class MigrationBuffer {
    public:
        virtual ~MigrationBuffer() = default;

        virtual void push(std::vector<Individual<GeneType>>&& incoming_deme_batch) = 0;

        virtual void popAll(std::vector<Individual<GeneType>>& target) = 0;

        [[nodiscard]] virtual bool hasMigrants() const = 0;
    };
}

#endif // MIGRATION_BUFFER_H