
#pragma once

#include "perceptron.hpp"

class Layer{
    private:
    std::vector<Perceptron> m_perceptrons;

    public:
    Layer()=default;
    ~Layer();

    const Perceptron& get_perceptron(uint index) const;
    void set_labels(const std::initializer_list<std::string>& labels);
    const std::string& label(uint index) const;
    void add_perceptron(uint n_perceptron=1);
    void insert_perceptron(uint index, const Perceptron& perceptron);
    void clear();
    unsigned get_nb_perceptrons() const;
    double ffeed_to(uint index);    // next layer perceptron index
};
