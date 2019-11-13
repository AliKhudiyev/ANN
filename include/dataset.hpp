
#pragma once

#include "matrix.hpp"

#include <map>

#define AUTOREAD    true
#define NAUTOREAD   false

class DataSet{
    private:
    Matrix m_inputSet, m_outputSet;
    std::string m_filePath;
    std::map<std::string, uint> m_map;

    public:
    DataSet()=default;
    DataSet(const std::string& filepath);
    ~DataSet();

    void load(const std::string& filepath, uint ignored_lines=0, bool auto_read=NAUTOREAD);
    void save(const std::string& filepath);
    void save();

    static Matrix one_hot_encode(double output, uint n);
    const Shape shape() const;
    void shuffle();
    std::vector<DataSet> split(const std::initializer_list<double>& percentages) const;
    DataSet split(uint beg, uint end) const;
    void print(uint n=1);
    void update();

    const std::vector<double>& get_input(uint index) const;
    const double get_output(uint index) const;
};
