#include "perceptron.hpp"

void Perceptron::set_input(double input){
    m_in=input;
}

double Perceptron::get_output() const{
    return m_out;
}

double Perceptron::mult(uint weight_index){
    return m_out*m_weight[weight_index][0];
}

double Perceptron::activate(){
    return m_activation(m_out);
}