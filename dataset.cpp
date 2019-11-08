#include "dataset.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>

DataSet::DataSet(const std::string& filepath){
    load(filepath);
}

DataSet::~DataSet(){}

void DataSet::load(const std::string& filepath, bool auto_read){
    m_filePath=filepath;

    std::stringstream stream;
    std::string line;
    std::ifstream file(filepath);

    uint dims[2], i=0, j=0;

    if(!auto_read){
        std::getline(file, line);
        stream<<line;
        while(std::getline(stream, line, ',')){
            dims[i++]=std::stoi(line);
        }
        m_inputSet.set_shape(dims[0], dims[1]-1);
        m_outputSet.set_shape(dims[0], 1);
        i=0;
        stream.clear();
    }

    // std::cout<<" dbg dims: "<<dims[0]<<", "<<dims[1]<<'\n';
    // std::cout<<" dbg input shape: "<<m_inputSet.shape()<<"; output shape: "<<m_outputSet.shape()<<'\n';

    while(std::getline(file, line)){
        stream<<line;
        while(std::getline(stream, line, ',')){
            if(j<dims[1]-1){
                // std::cout<<" dbg writing inputSet\n";
                m_inputSet.m_vals[i][j++]=strtod(line.c_str(), nullptr);
            }
            else{
                // std::cout<<" dbg writing outputSet\n";
                m_outputSet.m_vals[i++][0]=strtod(line.c_str(), nullptr);
                j=0;
            }
        }
        stream.clear();
    }

    file.close();
}

void DataSet::save(const std::string& filepath){
    ;
}

void DataSet::save(){
    save(m_filePath);
}

void DataSet::one_hot_encode(){
    ;
}

void DataSet::print(uint n){
    std::cout<<"Dataset ["<<m_inputSet.shape()<<"] - ["<<m_outputSet.shape()<<"]: "<<'\n';
    for(uint i=0;i<n;++i){
        for(uint j=0;j<m_inputSet.m_shape.n_col;++j){
            std::cout<<m_inputSet.m_vals[i][j]<<",\t\t";
        }
        std::cout<<m_outputSet.m_vals[i][0]<<'\n';
    }
}