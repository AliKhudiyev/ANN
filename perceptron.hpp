
#pragma once

#include "matrix.hpp"

#include <functional>

class Perceptron{
    private:
    double m_in, m_out;
    Matrix m_weight;
    std::function<double(double input)> m_activation;

    public:
    void set_input(double input);
    double get_output() const;
    double mult(uint weight_index);
    double activate();
};
