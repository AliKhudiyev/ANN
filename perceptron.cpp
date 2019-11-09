#include "perceptron.hpp"

#include <cstdlib>
#include <ctime>

Perceptron::~Perceptron(){}

void Perceptron::init_weights(const Shape& shape, double beg, double end) const{
    m_weight.set_shape(shape.n_row, shape.n_col);
    m_weight.random_init(beg, end);
}

void Perceptron::init_weights(uint n, double beg, double end) const{
    init_weights(Shape{n,1}, beg, end);
}

void Perceptron::set_input(double input){
    m_in=input;
}

double Perceptron::get_input() const{
    return m_in;
}

double Perceptron::get_output() const{
    return m_out;
}

void Perceptron::set_label(const std::string& label){
    m_label=label;
}

const std::string& Perceptron::label() const{
    return m_label;
}

double Perceptron::mult(uint weight_index){
    return m_out*m_weight[weight_index][0];
}

double Perceptron::activate(){
    return m_activation(m_out);
}