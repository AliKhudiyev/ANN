#include "ann.hpp"

#include <iostream>

ANN::ANN(uint n_layer, const std::initializer_list<uint>& n_perceptrons){
    auto it=n_perceptrons.begin();
    for(uint i=0;i<n_layer;++i, ++it){
        Layer layer;
        layer.add_perceptron(*it);
        m_layers.push_back(layer);
    }
}

ANN::~ANN(){}

void ANN::set_labels(LayerType label_type, const std::initializer_list<std::string>& labels){
    uint index;
    switch (label_type)
    {
    case InputLayer:
        index=0;
        break;
    case OutputLayer:
        index=m_layers.size()-1;
        break;
    default: break;
    }
    m_layers[index].set_labels(labels);
}

void ANN::set_dataSet(DataSet& data_set){
    m_dataSet=&data_set;
}

void ANN::set_batchSize(BatchSize batch_size){
    if(!m_dataSet){
        std::cout<<"ERROR [set_batchsize()]: No available dataset!\n";
    }
    
    Shape shape=m_dataSet->shape();
    switch (batch_size)
    {
    case None:
        ;
        break;
    case Half:
        m_batchSize=shape.n_row/2;
        break;
    case Quarter:
        m_batchSize=shape.n_row/4;
        break;
    default: break;
    }
}

void ANN::set_batchSize(uint batch_size){
    m_batchSize=batch_size;
}

void ANN::add_layer(const Layer& layer){
    m_layers.push_back(layer);
}

void ANN::add_layer(uint n_layer, const std::initializer_list<uint>& n_perceptrons){
    for(uint i=0;i<n_layer;++i){
    Layer layer;
    layer.add_perceptron(*(n_perceptrons.begin()+i));
    m_layers.push_back(layer);
    }
}

void ANN::insert_layer(uint index, const Layer& layer){
    m_layers.insert(m_layers.begin()+index, layer);
}

void ANN::insert_layer(uint index, uint n_perceptrons){
    Layer layer;
    layer.add_perceptron(n_perceptrons);
    insert_layer(index, layer);
}

void ANN::initialize(double beg, double end){
    uint n_row;
    uint n_perceptrons;

    for(uint i=0;i<m_layers.size()-1;++i){
        n_perceptrons=m_layers[i].get_nb_perceptrons();
        n_row=m_layers[i+1].get_nb_perceptrons();
        for(uint j=0;j<n_perceptrons;++j){
            m_layers[i].get_perceptron(j).init_weights(n_row, beg, end);
        }
    }
}

void ANN::train(DataSet& dataset, uint n_epoch, uint batch_size){
    m_dataSet=&dataset;
    if(batch_size!=-1) m_batchSize=batch_size;
    
    uint n_groups=m_dataSet->shape().n_row/m_batchSize;
    for(uint i=0;i<n_epoch;++i){
        for(uint j=0;j<n_groups;++j){
            for(uint k=0;k<m_batchSize;++k){
                ;
            }
        }
    }
}

void ANN::train(uint n_epoch, uint batch_size){
    train(*m_dataSet, n_epoch, batch_size);
}

std::string ANN::predict(std::vector<double>& inputs){
    uint index=m_layers.size()-1;
    Perceptron perceptron=m_layers[index].get_perceptron(0);
    double output=perceptron.get_output();

    for(uint i=1;i<m_layers[index].get_nb_perceptrons();++i){
        if(output<m_layers[index].get_perceptron(i).get_output()){
            perceptron=m_layers[index].get_perceptron(i);
            output=perceptron.get_output();
        }
    }
    return perceptron.label();
}

void ANN::print(uint index) const{
    Perceptron perceptron;
    for(uint i=0;i<m_layers[index].get_nb_perceptrons();++i){
        perceptron=m_layers[index].get_perceptron(i);
        std::cout<<m_layers[index].label(i)<<": "<<perceptron.get_output()<<'\n';
    }
}

void ANN::print(LayerType type) const{
    uint index;
    switch (type)
    {
    case InputLayer:
        index=0;
        break;
    case OutputLayer:
        index=m_layers.size()-1;
        break;
    default: break;
    }
    print(index);
}
void ANN::print_structure() const{
    uint last_index=m_layers.size()-1;
    std::cout<<" ||| Structure:\n";
    std::cout<<" > Number of layers: "<<last_index+1<<" |-";
    for(uint i=0;i<=last_index;++i){
        std::cout<<m_layers[i].get_nb_perceptrons()<<"-";
    }   std::cout<<"|\n";
    std::cout<<" > Input labels: ";
    for(uint i=0;i<m_layers[0].get_nb_perceptrons();++i){
        std::cout<<m_layers[0].label(i);
        if(i!=m_layers[0].get_nb_perceptrons()-1) std::cout<<", ";
    }
    std::cout<<"\n > Output labels: ";
    for(uint i=0;i<m_layers[last_index].get_nb_perceptrons();++i){
        std::cout<<m_layers[last_index].label(i);
        if(i!=m_layers[last_index].get_nb_perceptrons()-1) std::cout<<", ";
    }
    std::cout<<"\n= = = = = = = = = = = = = =\n";
}

void ANN::back_propagate(){
    ;
}