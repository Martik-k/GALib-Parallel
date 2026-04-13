#ifndef MPI_COMMUNICATOR_H
#define MPI_COMMUNICATOR_H

#pragma once

#include "algorithms/island/communication/communicators/Communicator.h"
#include "algorithms/island/communication/serializers/Serializer.h"
#include <mpi.h>
#include <vector>
#include <cstdint>

namespace galib {
    /**
     * @brief MPI-based implementation of the Communicator.
     * 
     * Uses non-blocking MPI primitives to enable asynchronous 
     * migration between processes.
     */
    template <typename GeneType>
    class MpiCommunicator : public Communicator<GeneType> {
    private:
        static constexpr int MPI_MESSAGE_TAG = 99;

        Serializer<GeneType>& serializer_m;
        std::vector<uint8_t> receive_buffer_m;
        MigrationBuffer<GeneType>* migration_buffer_m = nullptr;

        int rank_m;
        std::size_t network_size_m;

        bool is_running_m = false;
        MPI_Request mpi_request_m = MPI_REQUEST_NULL;

    public:
        explicit MpiCommunicator(Serializer<GeneType>& serializer, const std::size_t receive_buffer_size)
            : serializer_m(serializer), receive_buffer_m(receive_buffer_size) {
            int rank;
            int network_size;

            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &network_size);

            rank_m = rank;
            network_size_m = static_cast<std::size_t>(network_size);
        }

        ~MpiCommunicator() final {
            MpiCommunicator::stopReceiving();
        }

        void startReceiving(MigrationBuffer<GeneType>& target_buffer) override {
            stopReceiving();

            migration_buffer_m = &target_buffer;

            MPI_Irecv(
                receive_buffer_m.data(),
                static_cast<int>(receive_buffer_m.size()),
                MPI_BYTE,
                MPI_ANY_SOURCE,
                MPI_MESSAGE_TAG,
                MPI_COMM_WORLD,
                &mpi_request_m
            );

            is_running_m = true;
        }

        void update() override {
            if (!is_running_m) return;

            int is_received;

            while (true) {
                MPI_Status status;

                MPI_Test(
                    &mpi_request_m,
                    &is_received,
                    &status
                );

                if (!is_received) break;

                int received_bytes_count;

                MPI_Get_count(
                    &status,
                    MPI_BYTE,
                    &received_bytes_count
                );

                std::vector<Individual<GeneType>> migrants = serializer_m.deserialize(
                    receive_buffer_m.data(),
                    static_cast<std::size_t>(received_bytes_count)
                );

                migration_buffer_m->push(std::move(migrants));

                MPI_Irecv(
                    receive_buffer_m.data(),
                    static_cast<int>(receive_buffer_m.size()),
                    MPI_BYTE,
                    MPI_ANY_SOURCE,
                    MPI_MESSAGE_TAG,
                    MPI_COMM_WORLD,
                    &mpi_request_m
                );
            }
        }

        void stopReceiving() override {
            if (!is_running_m) return;

            is_running_m = false;

            if (mpi_request_m != MPI_REQUEST_NULL) {
                MPI_Cancel(&mpi_request_m);

                MPI_Status status;
                MPI_Wait(&mpi_request_m, &status);

                mpi_request_m = MPI_REQUEST_NULL;
            }
        }

        void send(const std::vector<Individual<GeneType>>& deme, const std::size_t destination_rank) override {
            const std::vector<std::uint8_t> serialized_data = serializer_m.serialize(deme);

            MPI_Send(
                serialized_data.data(),
                static_cast<int>(serialized_data.size()),
                MPI_BYTE,
                static_cast<int>(destination_rank),
                MPI_MESSAGE_TAG,
                MPI_COMM_WORLD
            );
        }

        void broadcast(const std::vector<Individual<GeneType>>& deme,
                       const std::vector<std::size_t>& destination_ranks) override {
            if (destination_ranks.empty()) return;

            const std::vector<std::uint8_t> serialized_data = serializer_m.serialize(deme);

            for (const auto& destination_rank : destination_ranks) {
                MPI_Send(
                    serialized_data.data(),
                    static_cast<int>(serialized_data.size()),
                    MPI_BYTE,
                    static_cast<int>(destination_rank),
                    MPI_MESSAGE_TAG,
                    MPI_COMM_WORLD
                );            }
        }

        Individual<GeneType> allReduceBest(Individual<GeneType> individual) const override {
            struct DoubleInt {
                double fitness;
                int rank;
            };

            const DoubleInt local_best{individual.getFitness(), rank_m};
            DoubleInt global_best;

            MPI_Allreduce(
                &local_best,
                &global_best,
                1,
                MPI_DOUBLE_INT,
                MPI_MINLOC,
                MPI_COMM_WORLD
            );

            std::vector<std::uint8_t> buffer;
            int buffer_size = 0;

            if (rank_m == global_best.rank) {
                buffer = std::move(serializer_m.serialize(individual));
                buffer_size = static_cast<int>(buffer.size());
            }

            MPI_Bcast(
                &buffer_size,
                1,
                MPI_INT,
                global_best.rank,
                MPI_COMM_WORLD
            );

            if (buffer_size > 0) {
                if (rank_m != global_best.rank) {
                    buffer.resize(buffer_size);
                }

                MPI_Bcast(
                    buffer.data(),
                    buffer_size,
                    MPI_BYTE,
                    global_best.rank,
                    MPI_COMM_WORLD
                );
            }

            if (rank_m == global_best.rank) { return individual; }

            auto deserialized = serializer_m.deserialize(buffer);
            if (deserialized.empty()) {
                return std::move(individual);
            }

            return std::move(deserialized[0]);
        }


        [[nodiscard]] std::size_t getRank() const override {
            return rank_m;
        }

        [[nodiscard]] std::size_t getSize() const override {
            return network_size_m;
        }
    };
}

#endif // MPI_COMMUNICATOR_H
