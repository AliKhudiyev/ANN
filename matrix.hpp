
#pragma once

#include <vector>
#include <string>

class DataSet;

struct Shape{
    union{
        uint m_dims[2];
        struct{
            uint n_row, n_col;
        };
    };

    bool requal(const Shape& shape) const;

    bool operator==(const Shape& shape) const;
    bool operator!=(const Shape& shape) const;

    friend std::ostream& operator<<(std::ostream& out, const Shape& shape);
};

class Matrix{
    friend class DataSet;

    private:
    std::vector<std::vector<double>> m_vals;
    Shape m_shape;

    public:
    Matrix()=default;
    Matrix(const Shape& shape);
    Matrix(uint n_row, uint n_col);
    ~Matrix();

    friend std::ostream& operator<<(std::ostream& out, const Matrix& matrix);
    static Matrix add(const Matrix& mat1, const Matrix& mat2);
    static Matrix sub(const Matrix& mat1, const Matrix& mat2);
    static Matrix dot(const Matrix& mat1, const Matrix& mat2);
    static Matrix mul(const Matrix& mat, double coef);
    static Matrix transpose(const Matrix& mat);
    const std::vector<double>& operator[](uint index) const;
    Matrix operator+(const Matrix& mat) const;
    Matrix operator-(const Matrix& mat) const;
    Matrix operator*(const Matrix& mat) const;
    Matrix& operator+=(const Matrix& mat);
    Matrix& operator-=(const Matrix& mat);
    Matrix& operator*=(const Matrix& mat);
    Matrix& mul(double coef);
    Matrix& transpose();

    double& set(uint n_row, uint n_col);
    void set(const std::vector<double>& vals, uint n_row);
    std::vector<double> get(uint n_row) const;
    void set_shape(uint n_row, uint n_col);
    const Shape& shape() const;
    void random_init(double beg=-0.01, double end=0.01);
};

std::ostream& operator<<(std::ostream& out, const Shape& shape);
std::ostream& operator<<(std::ostream& out, const Matrix& matrix);
