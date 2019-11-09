#include "layer.hpp"

Layer::~Layer(){}

const Perceptron& Layer::get_perceptron(uint index) const{
    return m_perceptrons[index];
}

void  Layer::set_labels(const std::initializer_list<std::string>& labels){
    auto label=labels.begin();
    for(uint i=0;i<m_perceptrons.size();++i, ++label){
        m_perceptrons[i].set_label(*label);
    }
}

const std::string& Layer::label(uint index) const{
    return m_perceptrons[index].label();
}

void Layer::add_perceptron(uint n_perceptron){
    for(uint i=0;i<n_perceptron;++i)
        m_perceptrons.push_back(Perceptron());
}

void Layer::insert_perceptron(uint index, const Perceptron& perceptron){
    m_perceptrons.insert(m_perceptrons.begin()+index, perceptron);
}

void Layer::clear(){
    m_perceptrons.clear();
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