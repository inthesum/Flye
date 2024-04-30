#include <cstdlib>
#include <stdexcept>
#include <cassert>
#include <vector>

template <class T>
class Matrix3d
{
public:
    Matrix3d(): _rows(0), _cols(0), _depth(0), _data(nullptr) {}
//    Matrix3d(const Matrix3d& other):
//            Matrix3d(other._rows, other._cols, other._depth)
//    {
//        for (size_t i = 0; i < _rows; ++i)
//            for (size_t j = 0; j < _cols; ++j)
//                for(size_t k = 0; k < _depth; ++k)
//                    this->at(i, j, k) = other.at(i, j, k);
//    }
    Matrix3d(Matrix3d&& other):
            Matrix3d()
    {
        std::swap(_rows, other._rows);
        std::swap(_cols, other._cols);
        std::swap(_depth, other._depth);
        std::swap(_data, other._data);
    }
    Matrix3d& operator=(Matrix3d && other)
    {
        std::swap(_rows, other._rows);
        std::swap(_cols, other._cols);
        std::swap(_depth, other._depth);
        std::swap(_data, other._data);
        return *this;
    }
    Matrix3d& operator=(const Matrix3d& other)
    {
        Matrix3d temp(other);
        std::swap(_rows, temp._rows);
        std::swap(_cols, temp._cols);
        std::swap(_depth, temp._depth);
        std::swap(_data, temp._data);
        return *this;
    }
    Matrix3d(size_t rows, size_t cols, size_t depth, T val = 0):
            _rows(rows), _cols(cols), _depth(depth)
    {
        if (!rows || !cols || !depth)
            throw std::runtime_error("Zero matrix dimension");
        _data = new T[rows * cols * depth];
    }
//    Matrix3d(T* data, size_t rows, size_t cols, size_t depth):
//            _data(data), _rows(rows), _cols(cols), _depth(depth)
//    {}
    ~Matrix3d()
    {
        if (_data) delete[] _data;
    }

    T& at(size_t row, size_t col, size_t depth) {return _data[row * _cols * _depth + col * _depth + depth];}
    const T& at(size_t row, size_t col, size_t depth) const {return _data[row * _cols * _depth + col * _depth + depth];}

    size_t nrows() const {return _rows;}
    size_t ncols() const {return _cols;}

    T* data() { return _data; }
    const T* data() const { return _data; }

private:
    size_t _rows;
    size_t _cols;
    size_t _depth;
    T* _data;
};
