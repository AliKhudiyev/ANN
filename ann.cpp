#include "ann.hpp"

#include "utils.cpp"

#include <iostream>
#include <cmath>

ANN::ANN(uint n_layer, const std::initializer_list<uint>& n_perceptrons){
    auto it=n_perceptrons.begin();
    m_n_input=*n_perceptrons.begin();
    m_n_output=*(n_perceptrons.end()-1);
    m_batchSize=-1;
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
    
    // std::cout<<" |> initialize\n";

    uint n_row;
    uint n_perceptrons;

    srand(time(0));
    for(uint i=0;i<m_layers.size()-1;++i){
        // std::cout<<" > layer "<<i+1<<'\n';
        n_perceptrons=m_layers[i].get_nb_perceptrons();
        n_row=m_layers[i+1].get_nb_perceptrons();

        // std::cout<<" # of perceptrons: "<<n_perceptrons<<'\n';
        // std::cout<<" # of next layer perceptrons: "<<n_row<<'\n';
        
        for(uint j=0;j<n_perceptrons;++j){
            // std::cout<<" >> perceptron "<<j+1<<'\n';
            m_layers[i].get_perceptron(j).init_weights(n_row, beg, end);
        }
    }

    // std::cout<<" :Done\n";

}

void ANN::train(DataSet& dataset, uint n_epoch, uint batch_size){

    // std::cout<<" |> train\n";
    // std::cout<<" n inputs: "<<m_n_input<<'\n';
    // for(uint i=0;i<m_layers[0].get_nb_perceptrons();++i){
        // std::cout<<"ouput for perceptron"<<i+1<<": "<<m_layers[0].get_perceptron(i).get_output()<<'\n';
        // std::cout<<"weight for perceptron "<<i+1<<":\n";
        // std::cout<<m_layers[0].get_perceptron(i).get_weights()<<'\n';
    // }

    m_dataSet=&dataset;
    double error;
    if(batch_size!=-1) m_batchSize=batch_size;
    else m_batchSize=m_dataSet->shape().n_row;
    
    // std::cout<<" > batch size: "<<m_batchSize<<'\n';

    uint n_groups=m_dataSet->shape().n_row/m_batchSize;

    // std::cout<<" > # groups: "<<n_groups<<"\n\n";

    for(uint i=0;i<n_epoch;++i){

        // std::cout<<" > epoch "<<i+1<<'\n';

        for(uint j=0;j<n_groups;++j){

            // std::cout<<" >> group "<<j+1<<'\n';

            for(uint k=0;k<m_batchSize;++k){

                // std::cout<<" >>> === sample index "<<k+1<<'\n';
                std::vector<double> inp=m_dataSet->get_input(j*batch_size+k);
                // std::cout<<" >>> inp: ("<<inp[0]<<", "<<inp[1]<<")\n";
                // std::cout<<" >>> network output: "<<net_output(m_dataSet->get_input(j*batch_size+k));
                // std::cout<<" >>> expected output: "<<DataSet::one_hot_encode(m_dataSet->get_output(j*batch_size+k), 2);

                // error=net_error(net_output(m_dataSet->get_input(j*m_batchSize+k))-DataSet::one_hot_encode(m_dataSet->get_output(j*m_batchSize+k), 3));
                back_propagate(net_output(m_dataSet->get_input(j*m_batchSize+k))-DataSet::one_hot_encode(m_dataSet->get_output(j*m_batchSize+k), 3), 0);

                // std::cout<<"bcs real ouput was "<<m_dataSet->get_output(j*m_batchSize+k)<<'\n';
                // std::cout<<" >>> error: "<<error<<'\n';
                // std::cout<<'\n';

            }
            // std::cout<<'\n';
        }
    }

    // std::cout<<" :Done\n";

}

void ANN::train(uint n_epoch, uint batch_size){
    train(*m_dataSet, n_epoch, batch_size);
}

std::string ANN::predict(const std::vector<double>& inputs){
    Matrix output=net_output(inputs);
    double max=0;
    uint index=0;

    for(uint i=0;i<output.shape().n_col;++i){
        if(max<output[0][i]){
            max=output[0][i];
            index=i;
        }
    }

    std::cout<<"Guessing: "<<output<<'\n';

    return m_layers[m_layers.size()-1].get_perceptron(index).label();
}

double ANN::accuracy(const DataSet& dataSet){
    uint right_guesses=0;
    uint all_guesses=dataSet.shape().n_row;

    for(uint i=0;i<all_guesses;++i){
        if(is_same(net_output(dataSet.get_input(i))[0], DataSet::one_hot_encode(dataSet.get_output(i), 3)[0]))
            ++right_guesses;
    }

    return (double)right_guesses/all_guesses;
}

void ANN::print(uint index) const{
    Perceptron perceptron;
    for(uint i=0;i<m_layers[index].get_nb_perceptrons();++i){
        const Perceptron perceptron=m_layers[index].get_perceptron(i);
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
        if(i!=m_layers[0].get_nb_perceptrons()-1 &&
           !m_layers[0].label(i+1).empty()) std::cout<<", ";
    }
    std::cout<<"\n > Output labels: ";
    for(uint i=0;i<m_layers[last_index].get_nb_perceptrons();++i){
        std::cout<<m_layers[last_index].label(i);
        if(i!=m_layers[last_index].get_nb_perceptrons()-1) std::cout<<", ";
    }
    std::cout<<"\n > Batch size: ";
    if(m_batchSize!=-1) std::cout<<m_batchSize;
    else    std::cout<<"none";
    std::cout<<"\n > Learning rate: "<<m_lr;
    if(m_dataSet){
        std::cout<<"\n > Dataset already exists (First 5 rows):\n";
        m_dataSet->print(5);
        
    }
    else{
        std::cout<<"\n > Dataset doesn't exist!\n";
    }
    std::cout<<"\n= = = = = = = = = = = = = =\n";
}

Matrix ANN::net_output(const std::vector<double>& inputs){
    m_layers[0].initialize(inputs);
    m_layers[0].get_perceptron(m_n_input-1).set_input(1);

    // std::cout<<"\n |> net_output\n";

    for(uint i=0;i<m_layers.size()-1;++i){

        // std::cout<<" > layer "<<i+1<<'\n';

        for(uint j=0;j<m_layers[i+1].get_nb_perceptrons();++j){

            // std::cout<<" >> perceptron "<<j+1<<'\n';

            m_layers[i+1].get_perceptron(j).set_input(m_layers[i].ffeed_to(j));
        }
    }
    
    // std::cout<<" |:Done\n";

    return m_layers[m_layers.size()-1].get();
}

double ANN::net_error(const Matrix& error) const{
    double err=0;

    for(uint i=0;i<error.shape().n_col;++i){
        err+=0.5*pow(error[0][i], 2);
    }

    return err;
}

void ANN::back_propagate(const Matrix& error, uint layer_index){
    for(uint i=0;i<error.shape().n_col;++i){
        for(uint j=0;j<m_layers[layer_index].get_nb_perceptrons();++j){
            m_layers[layer_index].get_perceptron(j).get_weights().set(i, 0)-=m_lr*error[0][i]*m_layers[layer_index].get_perceptron(j).get_output();
            // m_layers[layer_index].get_perceptron(j).get_output()
            // error[0][i]
        }
    }

}

void ANN::back_propagate(double error){
    // for(uint i=0;m_layers.size()-1;++i) back_propagate(error, m_layers.size()-2-i);
}