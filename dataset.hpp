
#pragma once

#include "matrix.hpp"

#define AUTOREAD    true
#define NAUTOREAD   false

class DataSet{
    private:
    Matrix m_inputSet, m_outputSet;
    std::string m_filePath;

    public:
    DataSet()=default;
    DataSet(const std::string& filepath);
    ~DataSet();

    void load(const std::string& filepath, bool auto_read=NAUTOREAD);
    void save(const std::string& filepath);
    void save();

    void one_hot_encode();
    const Shape shape() const;
    void print(uint n=1);
};
