#ifndef SERIALIZER_H
#define SERIALIZER_H

#pragma once

#include "core/Individual.h"
#include <vector>
#include <cstdint>

namespace galib {
    template <typename GeneType>
    class Serializer {
    public:
        virtual ~Serializer() = default;

        virtual std::vector<std::uint8_t> serialize(const std::vector<Individual<GeneType>>& individuals) const = 0;
        virtual std::vector<std::uint8_t> serialize(const Individual<GeneType>& individual) const = 0;

        virtual std::vector<Individual<GeneType>> deserialize(const std::vector<std::uint8_t>& individuals_serialized) const = 0;
        virtual std::vector<Individual<GeneType>> deserialize(const std::uint8_t* data, std::size_t size) const = 0;

        [[nodiscard]] virtual std::size_t getSerializedSize(std::size_t individuals_count, std::size_t genes_count) const = 0;
    };
}


#endif // SERIALIZER_H
