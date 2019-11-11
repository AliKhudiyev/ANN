
#pragma once

#include "layer.hpp"
#include "dataset.hpp"

#include <initializer_list>

// uint noBatchSize=-1;

enum BatchSize{
    None=0,
    Half,
    Quarter
};

enum LayerType{
    InputLayer=0,
    OutputLayer
};

typedef class ANN{
    const double learning_rate=0.0005;

    private:
    uint m_n_input, m_n_output;
    std::vector<Layer> m_layers;
    DataSet* m_dataSet=nullptr;
    uint m_batchSize=-1;
    double m_lr=learning_rate;

    public:
    ANN()=default;
    ANN(uint n_input, uint n_output=1): 
        ANN(2, {n_input+1, n_output}) {}
    ANN(uint n_layer, const std::initializer_list<uint>& n_perceptrons);
    ~ANN();

    void set_labels(LayerType label_type, const std::initializer_list<std::string>& labels);
    void set_dataSet(DataSet& data_set);
    void set_batchSize(BatchSize batch_size);
    void set_batchSize(uint batch_size);
    void add_layer(const Layer& layer);
    void add_layer(uint n_layer, const std::initializer_list<uint>& n_perceptrons);
    void insert_layer(uint index, const Layer& layer);
    void insert_layer(uint index, uint n_perceptrons);
    void initialize(double beg=-0.01, double end=0.01);
    void train(DataSet& dataset, uint n_epoch, uint batch_size=-1);
    void train(uint n_epoch, uint batch_size=-1);
    std::string predict(const std::vector<double>& inputs);
    double accuracy(const DataSet& dataSet);
    void print(uint index) const;
    void print(LayerType type) const;
    void print_structure() const;
    
    private:
    Matrix net_output(const std::vector<double>& inputs);
    double net_error(const Matrix& error) const;
    void back_propagate(const Matrix& error, uint layer_index);
    void back_propagate(double error);
}NNet, NeuralNet;
