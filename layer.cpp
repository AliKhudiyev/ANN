#include "layer.hpp"

void Layer::add_perceptron(uint n_perceptron){
    for(uint i=0;i<n_perceptron;++i)
        m_perceptrons.push_back(Perceptron());
}

void Layer::insert_perceptron(const std::initializer_list<uint>& indices){
    ;
}

unsigned Layer::get_nb_perceptrons() const{
    return m_perceptrons.size();
}

double Layer::ffeed_to(uint index){
    double weighted_sum=0.;

    for(uint i=0;i<m_perceptrons.size();++i){
        weighted_sum+=m_perceptrons[i].mult(index);
    }

    return weighted_sum;
}