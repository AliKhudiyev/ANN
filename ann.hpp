
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

typedef class ANN{
    private:
    uint m_n_input, m_n_output;
    std::vector<Layer> m_layers;
    DataSet* m_dataSet=nullptr;
    uint m_batchSize;

    public:
    ANN()=default;
    ANN(uint n_input, uint n_output=1);
    ANN(uint n_layer, const std::initializer_list<uint>& n_perceptrons);
    ~ANN();

    void set_batchSize(BatchSize batch_size);
    void set_batchSize(uint batch_size);
    void add_layer(const Layer& layer);
    void add_layer(uint n_layer, const std::initializer_list<uint>& n_perceptrons);
    void insert_layer(uint index, const Layer& layer);
    void insert_layer(uint index, uint n_perceptrons);
    void train(DataSet& dataset, uint n_epoch, uint batch_size=-1);
    void train(uint n_epoch, uint batch_size=-1);
    
    private:
    void back_propagate();
}NNet, NeuralNet;
