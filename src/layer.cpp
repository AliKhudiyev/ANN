#include "layer.hpp"

#include <iostream>

Layer::~Layer(){}

void Layer::initialize(const std::vector<double>& vals){
    for(uint i=0;i<m_perceptrons.size() && i<vals.size();++i){
        m_perceptrons[i].set_input(vals[i]);
    }
}

const Perceptron& Layer::get_perceptron(uint index) const{
    return m_perceptrons[index];
}

void  Layer::set_labels(const std::initializer_list<std::string>& labels){
    auto label=labels.begin();
    for(uint i=0;i<m_perceptrons.size() && label!=labels.end();++i, ++label){
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

Matrix Layer::get() const{
    Matrix mat(1, m_perceptrons.size());

    for(uint i=0;i<m_perceptrons.size();++i) mat.set(0, i)=m_perceptrons[i].get_output();

    return mat;
}