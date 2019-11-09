
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
    private:
    uint m_n_input, m_n_output;
    std::vector<Layer> m_layers;
    DataSet* m_dataSet=nullptr;
    uint m_batchSize;

    public:
    ANN()=default;
    ANN(uint n_input, uint n_output=1): 
        ANN(2, {n_input, n_output}) {}
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
    std::string predict(std::vector<double>& inputs);
    void print(uint index) const;
    void print(LayerType type) const;
    void print_structure() const;
    
    private:
    void back_propagate();
}NNet, NeuralNet;
