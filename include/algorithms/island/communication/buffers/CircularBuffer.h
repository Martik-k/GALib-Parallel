#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H

#pragma once

#include "algorithms/island/communication/buffers/MigrationBuffer.h"
#include "core/Individual.h"
#include <iterator>
#include <vector>
#include <mutex>
#include <atomic>

namespace galib::internal {
    template <typename GeneType>
    class CircularBuffer : public MigrationBuffer<GeneType> {
    private:
        const std::size_t max_batches_m;
        const std::size_t batch_size_m;
        std::atomic<std::size_t> filled_batches_m = 0;
        std::size_t head_batch_m = 0;

        std::vector<std::vector<Individual<GeneType>>> data_m;

        std::mutex mux_m;

    public:
        explicit CircularBuffer(const std::size_t max_batches, const std::size_t batch_size) : max_batches_m(max_batches),
            batch_size_m(batch_size),
            data_m(max_batches) {}

        void push(std::vector<Individual<GeneType>>&& incoming_deme_batch) override {
            {
                std::lock_guard<std::mutex> lock(mux_m);
                data_m[head_batch_m] = std::move(incoming_deme_batch);
                head_batch_m = (head_batch_m + 1) % max_batches_m;

                if (filled_batches_m < max_batches_m) {
                    ++filled_batches_m;
                }
            }
        }

        virtual void popAll(std::vector<Individual<GeneType>>& target) override {
            {
                std::lock_guard<std::mutex> lock(mux_m);
                target.clear();

                if (filled_batches_m == 0) { return; }

                target.reserve(batch_size_m * filled_batches_m.load());

                for (std::size_t i = 0; i < filled_batches_m; ++i) {
                    head_batch_m = (head_batch_m + max_batches_m - 1) % max_batches_m;

                    target.insert(target.end(),
                                  std::make_move_iterator(data_m[head_batch_m].begin()),
                                  std::make_move_iterator(data_m[head_batch_m].end()));

                    data_m[head_batch_m].clear();
                }

                filled_batches_m.store(0);
            }
        }

        [[nodiscard]] bool hasMigrants() const override {
            return filled_batches_m.load() > 0;
        }
    };
}

#endif // CIRCULAR_BUFFER_H
