
#pragma once

#include "matrix.hpp"

#include <functional>

class Perceptron{
    private:
    mutable double m_in, m_out;
    std::function<double(double input)> m_activation;
    std::string m_label;

    public:
    Perceptron();
    ~Perceptron();

    void set_input(double input) const;
    double get_input() const;
    double get_output() const;
    void set_label(const std::string& label);
    const std::string& label() const;
    void activate() const;

};
