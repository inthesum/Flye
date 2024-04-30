//#include <vector>
//#include <cstring> // Add this line for std::memset
//
//template <class T>
//class MemoryPool {
//public:
//    MemoryPool(size_t capacity) : _index(0), _capacity(capacity) {
//        T* start;
//        start = new T[capacity];
//        std::memset(start, 0, capacity * sizeof(T));
//        _start.push_back(start);
//        _usage.push_back(0);
//    }
//
//    ~MemoryPool() {
//        for (T* start : _start)
//            delete[] start;
//    }
//
//    T* allocate(size_t capacity) {
//        if (capacity > _capacity - _usage[_index]) {
//            if(_index + 1 >= _start.size()) {
//                std::cout << "Allocate new memory!" << std::endl;
//                T* start;
//                start = new T[_capacity];
//                std::memset(start, 0, _capacity * sizeof(T));
//                _start.push_back(start);
//                _usage.push_back(0);
//            }
//            _index++;
//        }
//
//        T* data = _start[_index] + _usage[_index];
//        _usage[_index] += capacity;
//        return data;
//    }
//
//    void reset() {
//        _index = 0;
//
//        for (T* start : _start)
//            std::memset(start, 0, _capacity * sizeof(T));
//
//        for(size_t& usage : _usage)
//            usage = 0;
//    }
//
//private:
//    size_t _index;
//    std::vector<T*> _start;
//    std::vector<size_t> _usage;
//    size_t _capacity;
//};


#include <vector>
#include <cstdlib> // Include this for malloc and free
#include <cstring> // Add this line for std::memset

template <class T>
class MemoryPool {
public:
    MemoryPool(size_t capacity) : _index(0), _capacity(capacity) {
        std::cout << "Allocate memory!" << std::endl;
        T* start;
        start = static_cast<T*>(std::malloc(capacity * sizeof(T)));
        if (!start) {
            throw std::bad_alloc(); // Handle allocation failure
        }
        _start.push_back(start);
        _usage.push_back(0);
    }

    ~MemoryPool() {
        for (T* start : _start)
            std::free(start);
    }

    T* allocate(size_t capacity) {
        if (capacity > _capacity - _usage[_index]) {
            if (_index + 1 >= _start.size()) {
                std::cout << "Allocate new memory!" << std::endl;
                T* start;
                start = static_cast<T*>(std::malloc(_capacity * sizeof(T)));
                if (!start) {
                    throw std::bad_alloc(); // Handle allocation failure
                }
                _start.push_back(start);
                _usage.push_back(0);
            }
            _index++;
        }

        T* data = _start[_index] + _usage[_index];
//        std::memset(data, 0, capacity * sizeof(T));
        _usage[_index] += capacity;
        return data;
    }

    void reset() {
        _index = 0;

        for (size_t& usage : _usage)
            usage = 0;
    }

private:
    size_t _index;
    std::vector<T*> _start;
    std::vector<size_t> _usage;
    size_t _capacity;
};
