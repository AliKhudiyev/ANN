
#pragma once

#include "perceptron.hpp"

class Layer{
    private:
    std::vector<Perceptron> m_perceptrons;

    public:
    void add_perceptron(uint n_perceptron=1);
    void insert_perceptron(const std::initializer_list<uint>& indices);
    unsigned get_nb_perceptrons() const;
    double ffeed_to(uint index);    // next layer perceptron index
};
