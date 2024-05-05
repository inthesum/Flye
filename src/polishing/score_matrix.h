//(c) 2016 by Authors
//This file is a part of ABruijn program.
//Released under the BSD license (see LICENSE file)

#include <cstdlib>
#include <stdexcept>
#include <cassert>
#include <vector>

#include "subs_matrix.h"

class ScoreMatrix
{
public:
	ScoreMatrix(): _rows(0), _cols(0), _data(nullptr) {}
	ScoreMatrix(ScoreMatrix&& other):
		ScoreMatrix()
	{
		std::swap(_cols, other._cols);
		std::swap(_rows, other._rows);
		std::swap(_data, other._data);
	}
	ScoreMatrix& operator=(ScoreMatrix && other)
	{
		std::swap(_cols, other._cols);
		std::swap(_rows, other._rows);
		std::swap(_data, other._data);
		return *this;
	}

    ScoreMatrix(size_t rows, size_t cols):
            _rows(rows), _cols(cols)
    {
        if (!rows || !cols)
            throw std::runtime_error("Zero matrix dimension");
        _data = new AlnScoreType[rows * cols];
    }
	~ScoreMatrix()
	{
		if (_data) delete[] _data;
	}

	AlnScoreType& at(size_t row, size_t col) {return _data[row * _cols + col];}
	const AlnScoreType& at(size_t row, size_t col) const {return _data[row * _cols + col];}
	size_t nrows() const {return _rows;}
	size_t ncols() const {return _cols;}

    AlnScoreType* data() { return _data; }
    const AlnScoreType* data() const { return _data; }

private:
	size_t _rows;
	size_t _cols;
	AlnScoreType* _data;
};
