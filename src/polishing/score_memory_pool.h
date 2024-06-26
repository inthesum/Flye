#pragma once

#include <cstdlib> // For std::malloc and std::free
#include <new> // For std::bad_alloc
#include <iostream> // For std::cerr
#include <xmmintrin.h> // For _mm_malloc and _mm_free

class ScoreMemoryPool {
public:
    ScoreMemoryPool(AlnScoreType* start, size_t capacity)
            : _start(start), _capacity(capacity), _usage(0) {
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

    void reset() {
        _usage = 0;
    }

private:
    size_t _capacity;
    size_t _usage;
    AlnScoreType* _start;
};


//#pragma once
//
//#include <cstdlib> // For std::malloc and std::free
//#include <new> // For std::bad_alloc
//#include <iostream> // For std::cout
//#include <xmmintrin.h> // For _mm_malloc and _mm_free
//
//class ScoreMemoryPool {
//public:
//    ScoreMemoryPool(size_t capacity) : _capacity(capacity), _usage(0) {
//        _start = (AlnScoreType*)_mm_malloc(capacity * sizeof(AlnScoreType), 32);
//        if (!_start) {
//            throw std::bad_alloc(); // Handle allocation failure
//        }
//    }
//
//    ~ScoreMemoryPool() {
//        _mm_free(_start);
//    }
//
//    AlnScoreType* allocate(size_t capacity) {
//        if (capacity > _capacity - _usage) {
//            std::cerr << "out of memory!" << std::endl;
//            return nullptr; // Not enough space, return nullptr
//        }
//
//        AlnScoreType* data = _start + _usage;
//        _usage += capacity;
//        return data;
//    }
//
//    void reset() {
//        _usage = 0;
//    }
//
//private:
//    size_t _capacity;
//    size_t _usage;
//    AlnScoreType* _start;
//};