
#pragma once

#include "perceptron.hpp"

class Layer{
    private:
    std::vector<Perceptron> m_perceptrons;
    mutable Matrix m_weights;

    public:
    Layer()=default;
    ~Layer();

    void initialize(const std::vector<double>& vals);
    void equalize(const std::vector<double>& inputs){
        for(uint i=0;i<m_perceptrons.size();++i){
            m_perceptrons[i].set_input(inputs[i]);
        }
    }
    const Perceptron& get_perceptron(uint index) const;
    void set_labels(const std::initializer_list<std::string>& labels);
    const std::string& label(uint index) const;
    void add_perceptron(uint n_perceptron=1);
    void insert_perceptron(uint index, const Perceptron& perceptron);
    void clear();
    unsigned get_nb_perceptrons() const;
    Matrix get() const;
    Matrix outputs() const{
        Matrix mat(1, m_perceptrons.size());
        for(uint i=0;i<m_perceptrons.size();++i)
            mat.set(0, i)=m_perceptrons[i].get_output();
        return mat;
    }
    Matrix inputs() const{
        Matrix mat(1, m_perceptrons.size());
        for(uint i=0;i<m_perceptrons.size();++i)
            mat.set(0, i)=m_perceptrons[i].get_input();
        return mat;
    }
    Matrix& weights() const{
        return m_weights;
    }
};
