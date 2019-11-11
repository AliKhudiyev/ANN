
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

    static Matrix one_hot_encode(double output, uint n);
    void one_hot_encode();
    const Shape shape() const;
    void shuffle();
    std::vector<DataSet> split(const std::initializer_list<double>& percentages) const;
    DataSet split(uint beg, uint end) const;
    void print(uint n=1);

    const std::vector<double>& get_input(uint index) const;
    const double get_output(uint index) const;
};
