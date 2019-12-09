
#pragma once

#include "layer.hpp"
#include "dataset.hpp"

#include <initializer_list>

// uint noBatchSize=-1;
struct Metric;

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
    const double learning_rate=0.01;
    friend class Metric;

    private:
    uint m_n_input, m_n_output;
    std::vector<Layer> m_layers;
    DataSet* m_dataSet=nullptr;
    uint m_batchSize=1;
    double m_lr=learning_rate;

    public:
    ANN()=default;
    ANN(uint n_input, uint n_output=1): 
        ANN({n_input+1, n_output}) {}
    ANN(const std::initializer_list<uint>& n_perceptrons);
    ~ANN();

    void set_labels(LayerType label_type, const std::initializer_list<std::string>& labels);
    void set_dataSet(DataSet& data_set);
    void set_batchSize(BatchSize batch_size);
    void set_batchSize(uint batch_size);
    void add_layer(const Layer& layer);
    void add_layer(uint n_layer, const std::initializer_list<uint>& n_perceptrons);
    void insert_layer(uint index, const Layer& layer);
    void insert_layer(uint index, uint n_perceptrons);
    void insert_layer(const std::initializer_list<uint>& indexes, const std::initializer_list<uint>& n_perceptrons);
    void insert_layer(const std::initializer_list<uint>& n_perceptrons);
    void initialize(double beg=-0.01, double end=0.01);
    void train(DataSet& dataset, uint n_epoch, uint batch_size=1);
    void train(uint n_epoch, uint batch_size=1);
    std::string predict(const std::vector<double>& inputs);
    double accuracy(const DataSet& dataSet);
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
    void print(uint index) const;
    void print(LayerType type) const;
    void print_structure() const;
    
    private:
    double net_error(const Matrix& error) const;
    Matrix feed_forward(const std::vector<double>& inputs);
    void back_propagate(const Matrix& error, uint layer_index);
    void back_propagate(const Matrix& error);
}NNet, NeuralNet;
