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

void Perceptron::activate() const{
    m_out=m_activation(m_in);
}