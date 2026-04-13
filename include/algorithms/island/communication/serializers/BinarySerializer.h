#ifndef BINARY_SERIALIZER_H
#define BINARY_SERIALIZER_H

#pragma once

#include "algorithms/island/communication/serializers/Serializer.h"
#include <vector>
#include <cstdint>
#include <bit>
#include <algorithm>
#include <type_traits>

namespace galib {
    template <typename GeneType>
    class BinarySerializer : public Serializer<GeneType> {
        static_assert(std::is_trivially_copyable_v<GeneType>, "GeneType must be trivially copyable for binary serialization");

    public:
        std::vector<std::uint8_t> serialize(const std::vector<Individual<GeneType>>& individuals) const override {
            if (individuals.empty()) { return {}; }

            const auto individuals_count = static_cast<std::uint32_t>(individuals.size());
            const auto individual_genes_count = static_cast<std::uint32_t>(individuals[0].getGenotype().size());

            const auto genotype_size_bytes = sizeof(GeneType) * individual_genes_count;

            const std::size_t total_size_bytes = (sizeof(std::uint32_t) * 2) + (individuals_count * (sizeof(double) + genotype_size_bytes));

            std::vector<std::uint8_t> data_serialized;
            data_serialized.reserve(total_size_bytes);

            appendSerialize(data_serialized, individuals_count);
            appendSerialize(data_serialized, individual_genes_count);

            for (const auto& individual : individuals) {
                appendSerialize(data_serialized, individual.getFitness());

                const auto& genotype = individual.getGenotype();

                if constexpr (std::endian::native == std::endian::little) {
                    const auto* data_ptr = reinterpret_cast<const std::uint8_t*>(genotype.data());
                    data_serialized.insert(data_serialized.end(), data_ptr, data_ptr + genotype_size_bytes);
                } else {
                    for (const auto& gene : genotype) {
                        appendSerialize(data_serialized, gene);
                    }
                }
            }

            return data_serialized;
        }

        std::vector<std::uint8_t> serialize(const Individual<GeneType>& individual) const override {
            const auto individual_genes_count = static_cast<std::uint32_t>(individual.getGenotype().size());

            const auto genotype_size_bytes = sizeof(GeneType) * individual_genes_count;

            const std::size_t total_size_bytes = (sizeof(std::uint32_t) * 2) + (sizeof(double) + genotype_size_bytes);

            std::vector<std::uint8_t> data_serialized;
            data_serialized.reserve(total_size_bytes);

            appendSerialize(data_serialized, static_cast<std::uint32_t>(1));
            appendSerialize(data_serialized, individual_genes_count);

            appendSerialize(data_serialized, individual.getFitness());

            const auto& genotype = individual.getGenotype();

            if constexpr (std::endian::native == std::endian::little) {
                const auto* data_ptr = reinterpret_cast<const std::uint8_t*>(genotype.data());
                data_serialized.insert(data_serialized.end(), data_ptr, data_ptr + genotype_size_bytes);
            } else {
                for (const auto& gene : genotype) {
                    appendSerialize(data_serialized, gene);
                }
            }

            return data_serialized;
        }

        std::vector<Individual<GeneType>> deserialize(const std::vector<std::uint8_t>& individuals_serialized) const override {
            if (individuals_serialized.empty()) {
                return {};
            }

            const std::uint8_t* read_ptr = individuals_serialized.data();

            const auto individuals_count = readDeserialize<uint32_t>(read_ptr);
            const auto individual_genes_count = readDeserialize<uint32_t>(read_ptr);

            std::vector<Individual<GeneType>> individuals;
            individuals.reserve(individuals_count);

            for (std::size_t i = 0; i < individuals_count; ++i) {
                const double fitness = readDeserialize<double>(read_ptr);

                std::vector<GeneType> genotype(individual_genes_count);

                if constexpr (std::endian::native == std::endian::little) {
                    const std::size_t genotype_bytes = sizeof(GeneType) * individual_genes_count;
                    std::copy_n(read_ptr, genotype_bytes, reinterpret_cast<std::uint8_t*>(genotype.data()));
                    read_ptr += genotype_bytes;
                } else {
                    for (std::size_t j = 0; j < individual_genes_count; ++j) {
                        genotype[j] = readDeserialize<GeneType>(read_ptr);
                    }
                }

                Individual<GeneType> individual;
                individual.setGenotype(std::move(genotype));
                individual.setFitness(fitness);
                individuals.push_back(std::move(individual));
            }

            return individuals;
        }

    private:
        template <typename T>
        static void appendSerialize(std::vector<uint8_t>& buffer, T value) {
            if constexpr (std::endian::native == std::endian::big) {
                auto* data_ptr = reinterpret_cast<std::uint8_t*>(&value);
                std::reverse(data_ptr, data_ptr + sizeof(T));
            }

            const auto bytes_ptr = reinterpret_cast<const std::uint8_t*>(&value);
            buffer.insert(buffer.end(), bytes_ptr, bytes_ptr + sizeof(T));
        }

        template <typename T>
        static T readDeserialize(const std::uint8_t*& ptr) {
            T value;
            std::copy_n(ptr, sizeof(T), reinterpret_cast<std::uint8_t*>(&value));
            ptr += sizeof(T);

            if constexpr (std::endian::native == std::endian::big) {
                auto* const data_ptr = reinterpret_cast<std::uint8_t*>(&value);
                std::reverse(data_ptr, data_ptr + sizeof(T));
            }

            return value;
        }
    };
}

#endif // BINARY_SERIALIZER_H
