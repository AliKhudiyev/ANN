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

Matrix DataSet::one_hot_encode(double output, uint n){
    Matrix encoded(1,n);

    encoded.set(0, (int)output)=1;

    return encoded;
}

void DataSet::one_hot_encode(){
    ;
}

const Shape DataSet::shape() const{
    Shape shape{m_inputSet.m_shape.n_row, m_inputSet.m_shape.n_col+1};
    return shape;
}

void DataSet::shuffle(){
    srand(time(0));
    uint n_row=m_inputSet.shape().n_row;
    std::vector<double> tmp;
    for(uint i=0;i<n_row;++i){
        uint i1=rand()%n_row, i2=rand()%n_row;
        tmp=m_inputSet[i1];
        m_inputSet.set(m_inputSet[i2], i1);
        m_inputSet.set(tmp, i2);

        tmp=m_outputSet[i1];
        m_outputSet.set(m_outputSet[i2], i1);
        m_outputSet.set(tmp, i2);
    }
}

std::vector<DataSet> DataSet::split(const std::initializer_list<double>& percentages) const{
    std::vector<DataSet> dataSets;
    DataSet tmp;
    uint beg=0, end=0;

    for(auto it=percentages.begin();it!=percentages.end();++it){
        end=beg+(*it)/100*m_inputSet.shape().n_row;
        tmp=split(beg, end);
        dataSets.push_back(tmp);
        beg=end;
    }

    return dataSets;
}

DataSet DataSet::split(uint beg, uint end) const{
    DataSet dataSet;

    dataSet.m_inputSet.set_shape(end-beg, m_inputSet.shape().n_col);
    dataSet.m_outputSet.set_shape(end-beg, m_inputSet.shape().n_col);
    for(uint i=0;i<end-beg;++i){
        for(uint j=0;j<m_inputSet.shape().n_col;++j){
            dataSet.m_inputSet.set(i, j)=m_inputSet[beg+i][j];
        }   dataSet.m_outputSet.set(i, 0)=m_outputSet[beg+i][0];
    }

    return dataSet;
}

void DataSet::print(uint n){
    std::cout<<"Dataset ["<<m_inputSet.shape()<<"] - ["<<m_outputSet.shape()<<"]: "<<'\n';
    for(uint i=0;i<n && i<m_inputSet.shape().n_row;++i){
        for(uint j=0;j<m_inputSet.m_shape.n_col;++j){
            std::cout<<m_inputSet.m_vals[i][j]<<",\t\t";
        }
        std::cout<<m_outputSet.m_vals[i][0]<<'\n';
    }
}

const std::vector<double>& DataSet::get_input(uint index) const{
    return m_inputSet[index];
}

const double DataSet::get_output(uint index) const{
    return m_outputSet[index][0];
}