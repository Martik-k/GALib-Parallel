#ifndef GRIDPOPULATION_H
#define GRIDPOPULATION_H


#pragma once
#include "Population.h"
#include <vector>
#include <utility>
#include <cstddef>
#include <stdexcept>


namespace galib {
template <typename GeneType>
class GridPopulation {
private:
    Population<GeneType> data_;  
    std::size_t rows_m;
    std::size_t cols_m;

public:
    GridPopulation(std::size_t rows, std::size_t cols, std::size_t num_genes)
        : data_(rows * cols, num_genes),
          rows_m(rows),
          cols_m(cols) {

        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("Grid dimensions cannot be zero");
        }
          }



    void initialize(GeneType lower, GeneType upper) {
        data_.initialize(lower, upper);
    }

    std::size_t size() const {
        return data_.size();
    }

    const Individual<GeneType>& getBestIndividual() const {
        return data_.getBestIndividual();
    }

    std::size_t index(std::size_t row, std::size_t col) const {
        if (row >= rows_m || col >= cols_m) {
            throw std::out_of_range("GridPopulation index out of range");
        }
        return row * cols_m + col;
    }

    Individual<GeneType>& at(std::size_t row, std::size_t col) {
        return data_[index(row, col)];
    }

    const Individual<GeneType>& at(std::size_t row, std::size_t col) const {
        return data_[index(row, col)];
    }

    std::size_t rows() const { return rows_m; }
    std::size_t cols() const { return cols_m; }

    std::size_t getNumGenes() const {
        return data_.getNumGenes();
    }


    std::vector<std::pair<std::size_t, std::size_t>>
    getNeighbors(std::size_t row, std::size_t col) const {
        std::vector<std::pair<std::size_t, std::size_t>> neighbors;

        if (row >= rows_m || col >= cols_m) {
            throw std::out_of_range("GridPopulation neighbor index out of range");
        }

       
        std::size_t up    = (row == 0) ? rows_m - 1 : row - 1;
        std::size_t down  = (row + 1) % rows_m;
        std::size_t left  = (col == 0) ? cols_m - 1 : col - 1;
        std::size_t right = (col + 1) % cols_m;

        neighbors.push_back({up, col});
        neighbors.push_back({down, col});
        neighbors.push_back({row, left});
        neighbors.push_back({row, right});

        return neighbors;
    }


};
}
