#ifndef SERIALIZER_H
#define SERIALIZER_H

#pragma once

#include "core/Individual.h"
#include <vector>
#include <cstdint>

namespace galib::internal {
    /**
     * @brief Interface for converting individuals to/from byte streams for network transport.
     * 
     * Implementations define the binary wire format used during migration.
     * 
     * @tparam GeneType The gene type of the individuals.
     */
    template <typename GeneType>
    class Serializer {
    public:
        virtual ~Serializer() = default;

        /**
         * @brief Serializes multiple individuals into a byte vector.
         * @param individuals The individuals to serialize.
         * @return A vector of bytes representing the individuals.
         */
        virtual std::vector<std::uint8_t> serialize(const std::vector<Individual<GeneType>>& individuals) const = 0;

        /**
         * @brief Serializes a single individual.
         * @param individual The individual to serialize.
         * @return A vector of bytes.
         */
        virtual std::vector<std::uint8_t> serialize(const Individual<GeneType>& individual) const = 0;

        /**
         * @brief Deserializes a byte vector into individuals.
         * @param individuals_serialized The byte stream to read.
         * @return A vector of reconstructed individuals.
         */
        virtual std::vector<Individual<GeneType>> deserialize(const std::vector<std::uint8_t>& individuals_serialized) const = 0;

        /**
         * @brief Deserializes from a raw pointer.
         * @param data Pointer to the start of the data.
         * @param size Number of bytes to read.
         * @return A vector of reconstructed individuals.
         */
        virtual std::vector<Individual<GeneType>> deserialize(const std::uint8_t* data, std::size_t size) const = 0;

        /**
         * @brief Calculates the exact size required to serialize a specific number of individuals.
         * @param individuals_count Number of individuals.
         * @param genes_count       Number of genes per individual.
         * @return Size in bytes.
         */
        [[nodiscard]] virtual std::size_t getSerializedSize(std::size_t individuals_count, std::size_t genes_count) const = 0;
    };
}


#endif // SERIALIZER_H
