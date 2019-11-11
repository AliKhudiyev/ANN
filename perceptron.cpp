#include "perceptron.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>

double def_act_func(double input){
    return input;
}

Perceptron::Perceptron(){
    m_activation=def_act_func;
}

Perceptron::~Perceptron(){}

void Perceptron::init_weights(const Shape& shape, double beg, double end) const{
    
    // std::cout<<" |> init_weights\n";
    // std::cout<<" > shape: "<<shape<<'\n';

    m_weight.set_shape(shape.n_row, shape.n_col);

    // std::cout<<" > setting shape\n";

    m_weight.random_init(beg, end);

    // std::cout<<" > random initializement\n";
    // std::cout<<" :Done\n";
}

void Perceptron::init_weights(uint n, double beg, double end) const{
    init_weights(Shape{n,1}, beg, end);
}

Matrix& Perceptron::get_weights() const{
    return m_weight;
}

void Perceptron::set_input(double input) const{
    m_in=input;
    activate();
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

    // std::cout<<" www: "<<m_weight[weight_index][0]<<'\n';

    return m_out*m_weight[weight_index][0];
}

void Perceptron::activate() const{
    m_out=m_activation(m_in);
}