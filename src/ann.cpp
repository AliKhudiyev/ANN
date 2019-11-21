#include "ann.hpp"

#include "utils.cpp"

#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>

ANN::ANN(const std::initializer_list<uint>& n_perceptrons){
    auto it=n_perceptrons.begin();
    m_n_input=*n_perceptrons.begin();
    m_n_output=*(n_perceptrons.end()-1);
    m_batchSize=-1;
    for(auto it=n_perceptrons.begin();it!=n_perceptrons.end();++it){
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

void ANN::insert_layer(const std::initializer_list<uint>& indexes, const std::initializer_list<uint>& n_perceptrons){
    if(indexes.size()==n_perceptrons.size()){
        auto index_it=indexes.begin();
        auto percept_it=n_perceptrons.begin();

        for(;index_it!=indexes.end();++index_it, ++percept_it){
            insert_layer(*index_it, *percept_it);
        }
    }
}

void ANN::insert_layer(const std::initializer_list<uint>& n_perceptrons){
    uint i=0;
    for(auto it=n_perceptrons.begin();it!=n_perceptrons.end();++it){
        insert_layer(i++, *it);
    }
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
    m_dataSet=&dataset;
    double error;
    if(batch_size==0) m_batchSize=m_dataSet->shape().n_row;
    m_batchSize=batch_size;
    uint n_groups=m_dataSet->shape().n_row/m_batchSize;

    for(uint i=0;i<n_epoch;++i){
        for(uint j=0;j<n_groups;++j){
            for(uint k=0;k<m_batchSize;++k){
                std::vector<double> inp=m_dataSet->get_input(j*batch_size+k);
                back_propagate(net_output(m_dataSet->get_input(j*m_batchSize+k))-DataSet::one_hot_encode(m_dataSet->get_output(j*m_batchSize+k), 2));
            }
        }
    }
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

/*
 * Weight file structure: 
 * -----------------------
 * # of layers
 * # of perceptrons in layer 1
 * # of weights in perceptron 1 of the current layer (layer 1)
 * weights
*/

void ANN::save(const std::string& filepath) const{
    uint n_perceptron=0, n_weights=0;
    std::ofstream file(filepath);

    file<<m_layers.size()-1<<"\n";
    for(uint i=0;i<m_layers.size()-1;++i){
        n_perceptron=m_layers[i].get_nb_perceptrons();
        file<<n_perceptron<<"\n";
        n_weights=m_layers[i+1].get_nb_perceptrons();
        file<<n_weights<<"\n";
        // std::cout<<"layer: "<<i<<", "<<n_perceptron<<'\n';
        for(uint j=0;j<n_perceptron;++j){
            // file<<"perceptron, "<<j<<"\n";
            // std::cout<<"perceptron: "<<j<<'\n';
            // file<<m_layers[i].get_perceptron(j).get_weights();
            m_layers[i].get_perceptron(j).get_weights().print_weights(file);
            file<<'\n';
        }
    }

    file.close();
}

void ANN::load(const std::string& filepath){
    std::stringstream stream;
    std::string line;
    uint n_perceptron, n_layer, n_weights;
    Matrix weight;
    // std::vector<double> vals(1);

    std::ifstream file(filepath);

    if(!file){
        std::cout<<"ERROR [weight file]: Couldn't load weights!\n";
    }

    std::getline(file, line);
    n_layer=std::stoi(line);

    if(n_layer!=m_layers.size()-1){
        std::cout<<"ERROR [weight loading]: Inaccurate # of layers!\n";
    }

    for(uint i=0;i<n_layer;++i){
        // std::cout<<"layer "<<i+1<<'\n';

        std::getline(file, line);
        n_perceptron=std::stoi(line);
        std::getline(file, line);
        n_weights=std::stoi(line);
        weight.set_shape(n_weights, 1);
        
        if(n_perceptron!=m_layers[i].get_nb_perceptrons()){
            std::cout<<"ERROR [weight loading]: Inaccurate # of perceptrons of layer "<<i+1<<'\n';
        }

        for(uint j=0;j<n_perceptron;++j){
            // std::cout<<"perceptron "<<j+1<<'\n';
            std::getline(file, line);
            stream.clear();
            stream<<line;
            for(uint k=0;k<n_weights;++k){
                std::getline(stream, line, ',');
                // std::cout<<" > "<<std::stold(line)<<'\n';
                // vals[0]=std::stold(line);
                weight.set(k, 0)=std::stold(line);
            }
            m_layers[i].get_perceptron(j).set_weights(weight);
            // m_layers[i].get_perceptron(j).get_weights().print_weights(std::cout)<<'\n';
            // std::cout<<'\n';
        }
    }

    file.close();
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
    std::cout<<" > Input labels:\n\t";
    for(uint i=0;i<m_layers[0].get_nb_perceptrons();++i){
        std::cout<<m_layers[0].label(i);
        if(i!=m_layers[0].get_nb_perceptrons()-1 &&
           !m_layers[0].label(i+1).empty()) std::cout<<"\n\t";
    }
    std::cout<<"\n > Output labels:\n\t";
    for(uint i=0;i<m_layers[last_index].get_nb_perceptrons();++i){
        std::cout<<m_layers[last_index].label(i);
        if(i!=m_layers[last_index].get_nb_perceptrons()-1) std::cout<<"\n\t";
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
        std::cout<<": "<<error[0][i]<<" :";
        err+=0.5*pow(error[0][i], 2);
    }   std::cout<<'\n';

    return err;
}

void ANN::back_propagate(const Matrix& error, uint layer_index){
    // std::cout<<"Error: ";
    for(uint i=0;i<error.shape().n_col;++i){
        // std::cout<<error[0][i]<<" \n";
        for(uint j=0;j<m_layers[layer_index].get_nb_perceptrons();++j){
            double sign=1;
            if(error[0][i]<0) sign*=-1;
            m_layers[layer_index].get_perceptron(j).get_weights().set(i, 0)-=m_lr*sign*m_layers[layer_index].get_perceptron(j).get_output();
            // m_layers[layer_index].get_perceptron(j).get_output()
            // error[0][i]
        }
    }
    // std::cout<<'\n';

}

void ANN::back_propagate(const Matrix& error){
    // for(uint i=0;m_layers.size()-1;++i) back_propagate(error, m_layers.size()-2-i);
    for(uint i=0;i<m_layers.size()-1;++i){
        back_propagate(error, i);
    }
}