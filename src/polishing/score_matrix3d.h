#include <stdexcept>

#include <immintrin.h>
#include "subs_matrix.h"

class ScoreMatrix3d
{
public:
    ScoreMatrix3d(): _rows(0), _cols(0), _depth(0), _data(nullptr) {}
    ScoreMatrix3d(ScoreMatrix3d&& other):
            ScoreMatrix3d()
    {
        std::swap(_rows, other._rows);
        std::swap(_cols, other._cols);
        std::swap(_depth, other._depth);
        std::swap(_data, other._data);
    }
    ScoreMatrix3d& operator=(ScoreMatrix3d && other)
    {
        std::swap(_rows, other._rows);
        std::swap(_cols, other._cols);
        std::swap(_depth, other._depth);
        std::swap(_data, other._data);
        return *this;
    }

    ScoreMatrix3d(AlnScoreType* data, size_t rows, size_t cols, size_t depth):
        _data(data), _rows(rows), _cols(cols), _depth(depth) {}
    ScoreMatrix3d(size_t rows, size_t cols, size_t depth):
            _rows(rows), _cols(cols), _depth(depth)
    {
        if (!rows || !cols || !depth)
            throw std::runtime_error("Zero matrix dimension");
        _data = (AlnScoreType*)_mm_malloc(rows * cols * depth * sizeof(AlnScoreType), 64);
    }
    ~ScoreMatrix3d()
    {
        if (_data) _mm_free(_data);
    }

    AlnScoreType& at(size_t row, size_t col, size_t depth) {return _data[row * _cols * _depth + col * _depth + depth];}
    const AlnScoreType& at(size_t row, size_t col, size_t depth) const {return _data[row * _cols * _depth + col * _depth + depth];}

    size_t nrows() const {return _rows;}
    size_t ncols() const {return _cols;}

    AlnScoreType* data() { return _data; }
    const AlnScoreType* data() const { return _data; }

private:
    size_t _rows;
    size_t _cols;
    size_t _depth;
    AlnScoreType* _data;
};
