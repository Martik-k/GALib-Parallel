#ifndef GRIDPOPULATION_H
#define GRIDPOPULATION_H


#pragma once
#include "Population.h"
#include <vector>
#include <utility>
#include <cstddef>
#include <stdexcept>


namespace galib {
/**
 * @brief A population organized in a 2D grid structure.
 * 
 * Primarily used by Cellular Genetic Algorithms where spatial locality 
 * and neighborhood relationships are important for selection.
 * 
 * @tparam GeneType The type of the genes.
 */
template <typename GeneType>
class GridPopulation {
private:
    Population<GeneType> data_;  
    std::size_t rows_m;
    std::size_t cols_m;

public:
    /**
     * @brief Constructs a grid population with specified dimensions.
     * @param rows      Number of rows in the grid.
     * @param cols      Number of columns in the grid.
     * @param num_genes Number of genes each individual should have.
     * @throws std::invalid_argument if dimensions are zero.
     */
    GridPopulation(std::size_t rows, std::size_t cols, std::size_t num_genes)
        : data_(rows * cols, num_genes),
          rows_m(rows),
          cols_m(cols) {

        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("Grid dimensions cannot be zero");
        }
    }

    /**
     * @brief Randomly initializes all individuals in the grid.
     * @param lower Lower bound for genes.
     * @param upper Upper bound for genes.
     */
    void initialize(GeneType lower, GeneType upper) {
        data_.initialize(lower, upper);
    }

    /**
     * @brief Returns the total number of individuals in the grid.
     * @return rows * cols.
     */
    std::size_t size() const {
        return data_.size();
    }

    /**
     * @brief Returns the best individual found in the entire grid.
     * @return Constant reference to the best individual.
     */
    const Individual<GeneType>& getBestIndividual() const {
        return data_.getBestIndividual();
    }

    /**
     * @brief Converts 2D grid coordinates to a 1D linear index.
     * @param row Grid row.
     * @param col Grid column.
     * @return Linear index in the underlying population.
     * @throws std::out_of_range if coordinates are invalid.
     */
    std::size_t index(std::size_t row, std::size_t col) const {
        if (row >= rows_m || col >= cols_m) {
            throw std::out_of_range("GridPopulation index out of range");
        }
        return row * cols_m + col;
    }

    /**
     * @brief Accesses an individual at specific grid coordinates.
     * @param row Grid row.
     * @param col Grid column.
     * @return Reference to the individual.
     */
    Individual<GeneType>& at(std::size_t row, std::size_t col) {
        return data_[index(row, col)];
    }

    /**
     * @brief Accesses an individual at specific grid coordinates (const version).
     * @param row Grid row.
     * @param col Grid column.
     * @return Constant reference to the individual.
     */
    const Individual<GeneType>& at(std::size_t row, std::size_t col) const {
        return data_[index(row, col)];
    }

    /** @brief Returns the number of rows. */
    std::size_t rows() const { return rows_m; }

    /** @brief Returns the number of columns. */
    std::size_t cols() const { return cols_m; }

    /**
     * @brief Returns the number of genes for individuals in this population.
     * @return Gene count.
     */
    std::size_t getNumGenes() const {
        return data_.getNumGenes();
    }

    /**
     * @brief Accesses an individual via its linear index.
     * @param index Linear index.
     * @return Reference to the individual.
     */
    Individual<GeneType>& linearAt(std::size_t index) {
        if (index >= data_.size()) {
            throw std::out_of_range("GridPopulation linear index out of range");
        }
        return data_[index];
    }

    /**
     * @brief Accesses an individual via its linear index (const version).
     * @param index Linear index.
     * @return Constant reference to the individual.
     */
    const Individual<GeneType>& linearAt(std::size_t index) const {
        if (index >= data_.size()) {
            throw std::out_of_range("GridPopulation linear index out of range");
        }
        return data_[index];
    }

    /** @brief Checks if the population is empty. */
    bool empty() const {
        return data_.empty();
    }

    /**
     * @brief Retrieves the 4-neighborhood (Up, Down, Left, Right) of a cell.
     * 
     * Uses periodic (toroidal) boundary conditions.
     * 
     * @param row Cell row.
     * @param col Cell column.
     * @return Vector of (row, col) pairs representing neighbors.
     */
    std::vector<std::pair<std::size_t, std::size_t>>
    getNeighbors(std::size_t row, std::size_t col) const {        std::vector<std::pair<std::size_t, std::size_t>> neighbors;

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
#endif // GRIDPOPULATION_H
