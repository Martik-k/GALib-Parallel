#ifndef FITNESS_FUNCTION_H
#define FITNESS_FUNCTION_H

#pragma once

namespace galib {

    template <typename GeneType = double>
    class FitnessFunction {
    public:
        virtual ~FitnessFunction() = default;

        virtual double evaluate(const std::vector<GeneType>& phenotype) const = 0;
        virtual std::size_t size() const = 0;
        virtual GeneType getLowerBound(std::size_t dimension) const = 0;
        virtual GeneType getUpperBound(std::size_t dimension) const = 0;
    };
}

#endif // FITNESS_FUNCTION_H
