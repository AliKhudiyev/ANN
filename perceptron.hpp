
#pragma once

#include "matrix.hpp"

#include <functional>

class Perceptron{
    private:
    mutable double m_in, m_out;
    mutable Matrix m_weight;
    std::function<double(double input)> m_activation;
    std::string m_label;

    public:
    Perceptron();
    ~Perceptron();

    void init_weights(const Shape& shape, double beg=-0.01, double end=0.01) const;
    void init_weights(uint n, double beg=-0.01, double end=0.01) const;
    Matrix& get_weights() const;
    void set_weights(const Matrix& weight) const;
    void set_input(double input) const;
    double get_input() const;
    double get_output() const;
    void set_label(const std::string& label);
    const std::string& label() const;
    double mult(uint weight_index);
    void activate() const;

};
