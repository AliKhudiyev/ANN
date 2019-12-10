#include "matrix.hpp"

#include <iostream>
#include <cstdlib>
#include <ctime>

bool Shape::requal(const Shape& shape) const{
    return (n_row==shape.n_col && n_col==shape.n_row);
}

bool Shape::operator==(const Shape& shape) const{
    return (n_row==shape.n_row && n_col==shape.n_col);
}

bool Shape::operator!=(const Shape& shape) const{
    return !(*this==shape);
}

std::ostream& operator<<(std::ostream& out, const Shape& shape){
    out<<shape.n_row<<", "<<shape.n_col;
    return out;
}

std::ostream& operator<<(std::ostream& out, const Matrix& matrix){
    for(uint i=0;i<matrix.m_shape.n_row;++i){
        for(uint j=0;j<matrix.m_shape.n_col;++j){
            out<<matrix[i][j]<<"\t";
        }   out<<'\n';
    }

    return out;
}

Matrix::Matrix(const Shape& shape){
    set_shape(shape.n_row, shape.n_col);
}

Matrix::Matrix(uint n_row, uint n_col){
    set_shape(n_row, n_col);
}

Matrix::~Matrix(){}

Matrix Matrix::add(const Matrix& mat1, const Matrix& mat2){
    return mat1+mat2;
}

Matrix Matrix::sub(const Matrix& mat1, const Matrix& mat2){
    return mat1-mat2;
}

Matrix Matrix::dot(const Matrix& mat1, const Matrix& mat2){
    return mat1*mat2;
}

Matrix Matrix::mul(const Matrix& mat, double coef){
    Matrix result=mat;
    
    for(auto& vec: result.m_vals){
        for(auto& val: vec){
            val*=coef;
        }
    }
    
    return result;
}

Matrix Matrix::mulew(const Matrix& mat1, const Matrix& mat2){
    Matrix mat=mat1;
    return mat.mulew(mat2);
}

Matrix Matrix::transpose(const Matrix& mat){
    Matrix result(mat.m_shape.n_col, mat.m_shape.n_row);

    for(uint i=0;i<mat.m_shape.n_row;++i){
        for(uint j=0;j<mat.m_shape.n_col;++j){
            result.m_vals[j][i]=mat.m_vals[i][j];
        }
    }
    return result;
}

const std::vector<double>& Matrix::operator[](uint index) const{
    if(index>=m_shape.n_row){
        std::cout<<"ERROR [[] operator]: Index out of bounds!\n";
        exit(1);
    }

    return m_vals[index];
}

Matrix Matrix::operator+(const Matrix& mat) const{
    if(m_shape!=mat.m_shape){
        std::cout<<"ERROR [+ operator]: Shape doesn't match!\n";
        exit(1);
    }

    Matrix result(m_shape);
    
    for(uint i=0;i<m_shape.n_row;++i){
        for(uint j=0;j<m_shape.n_col;++j){
            result.m_vals[i][j]=m_vals[i][j]+mat.m_vals[i][j];
        }
    }
    
    return result;
}

Matrix Matrix::operator-(const Matrix& mat) const{
    if(m_shape!=mat.m_shape){
        std::cout<<"ERROR [- operator]: Shape doesn't match!\n";
        exit(1);
    }

    Matrix result(m_shape);
    
    for(uint i=0;i<m_shape.n_row;++i){
        for(uint j=0;j<m_shape.n_col;++j){
            result.m_vals[i][j]=m_vals[i][j]-mat.m_vals[i][j];
        }
    }
    
    return result;
}

Matrix Matrix::operator*(const Matrix& mat) const{
    if(m_shape.n_col!=mat.m_shape.n_row){
        std::cout<<"ERROR [* operator]: Shape doesn't match!\n";
        exit(1);
    }
    Matrix result(m_shape.n_row, mat.m_shape.n_col);
    
    for(uint i=0;i<m_shape.n_row;++i){
        for(uint t=0;t<mat.m_shape.n_col;++t){
            result.m_vals[i][t]=0;
            for(uint j=0;j<m_shape.n_col;++j){
                result.m_vals[i][t]+=m_vals[i][j]*mat.m_vals[j][t];
            }
        }
    }
    // std::cout<<"+\n";
    return result;
}

Matrix& Matrix::operator+=(const Matrix& mat){
    *this=(*this)+mat;
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& mat){
    *this=(*this)-mat;
    return *this;
}

Matrix& Matrix::operator*=(const Matrix& mat){
    *this=(*this)*mat;
    return *this;
}

Matrix& Matrix::mul(double coef){
    for(auto& vec: m_vals){
        for(auto& val: vec){
            val*=coef;
        }
    }

    return *this;
}

Matrix& Matrix::mulew(const Matrix& mat){
    if(m_shape!=mat.m_shape){
        std::cout<<"ERROR [element-wise multiplication]: Shapes don't match!\n";
        exit(1);
    }

    for(uint i=0;i<m_shape.n_row;++i){
        for(uint j=0;j<m_shape.n_col;++j){
            m_vals[i][j]*=mat.m_vals[i][j];
        }
    }
    return *this;
}

Matrix& Matrix::transpose(){
    Matrix transposed(m_shape.n_col, m_shape.n_row);
    // double tmp;

    for(uint i=0;i<m_shape.n_col;++i){
        for(uint j=0;j<m_shape.n_row;++j){
            // tmp=m_vals[i][j];
            transposed.m_vals[i][j]=m_vals[j][i];
        }
    }
    m_vals=transposed.m_vals;

    return *this;
}

double& Matrix::set(uint n_row, uint n_col){
    return m_vals[n_row][n_col];
}

void Matrix::set(const std::vector<double>& vals, uint n_row){
    m_vals[n_row]=vals;
}

std::vector<double> Matrix::get(uint n_row) const{
    return m_vals[n_row];
}

void Matrix::set_shape(uint n_row, uint n_col){
    static uint count=0;
    if(m_shape!=Shape{0,0}){
        for(auto& vec: m_vals)
            vec.clear();
        m_vals.clear();
    }

    m_shape.n_row=n_row;
    m_shape.n_col=n_col;

    m_vals.resize(n_row);
    for(auto& vec: m_vals){
        vec.resize(n_col);
        for(auto& val: vec)
            val=0.;
    }
}

const Shape& Matrix::shape() const{
    return m_shape;
}

void Matrix::random_init(double beg, double end){
    for(auto& vec: m_vals){
        for(auto& val: vec){
            val=(end-beg)*(double)rand()/RAND_MAX+beg;
        }
    }
}

std::ostream& Matrix::print_weights(std::ostream& out) const{
    for(uint i=0;i<m_shape.n_row;++i){
        for(uint j=0;j<m_shape.n_col;++j){
            out<<m_vals[i][j]<<", ";
        }   out<<'\n';
    }
    return out;
}