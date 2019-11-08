#include "ann.hpp"

ANN::ANN(uint n_input, uint n_output){
    ;
}

ANN::ANN(uint n_layer, const std::initializer_list<uint>& n_perceptrons){
    ;
}

ANN::~ANN(){}

void ANN::set_batchSize(BatchSize batch_size){
    ;
}

void ANN::set_batchSize(uint batch_size){
    ;
}

void ANN::add_layer(const Layer& layer){
    ;
}

void ANN::add_layer(uint n_layer, const std::initializer_list<uint>& n_perceptrons){
    ;
}

void ANN::insert_layer(uint index, const Layer& layer){
    ;
}

void ANN::insert_layer(uint index, uint n_perceptrons){
    ;
}

void ANN::train(DataSet& dataset, uint n_epoch, uint batch_size){
    for(uint i=0;i<n_epoch;++i){
        ;
    }
}

void ANN::train(uint n_epoch, uint batch_size){
    ;
}

void ANN::back_propagate(){
    ;
}