#pragma once

#include <cstdlib> // For std::malloc and std::free
#include <new> // For std::bad_alloc
#include <iostream> // For std::cerr
#include <xmmintrin.h> // For _mm_malloc and _mm_free

class ScoreMemoryPool {
public:
    ScoreMemoryPool(AlnScoreType* start, size_t capacity)
            : _start(start), _capacity(capacity), _usage(0), _boundary(0) {
        if (!_start) {
            throw std::bad_alloc(); // Handle allocation failure
        }
    }

    AlnScoreType* allocate(size_t capacity) {
        if (capacity > _capacity - _usage) {
            std::cerr << "out of memory!" << std::endl;
            return nullptr; // Not enough space, return nullptr
        }

        AlnScoreType* data = _start + _usage;
        _usage += capacity;
        return data;
    }

    void set_boundary() {
        _boundary = _usage;
    }

    void reset_to_boundary() {
        _usage = _boundary;
    }

    void reset() {
        _boundary = 0;
        _usage = 0;
    }

private:
    size_t _capacity;
    size_t _usage;
    size_t _boundary;
    AlnScoreType* _start;
};