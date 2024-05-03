#include <vector>
#include <cstdlib> // Include this for malloc and free
#include <cstring> // Add this line for std::memset

class ScoreMemoryPool {
public:
    ScoreMemoryPool(size_t capacity) : _index(0), _capacity(capacity) {
        std::cout << "Allocate memory!" << std::endl;
        AlnScoreType* start;
//        start = static_cast<AlnScoreType*>(std::malloc(capacity * sizeof(AlnScoreType)));
        start = (AlnScoreType*)_mm_malloc(capacity * sizeof(AlnScoreType), 32);
        if (!start) {
            throw std::bad_alloc(); // Handle allocation failure
        }
        _start.push_back(start);
        _usage.push_back(0);
    }

    ~ScoreMemoryPool() {
        for (AlnScoreType* start : _start)
            _mm_free(start);
//            std::free(start);
    }

    AlnScoreType* allocate(size_t capacity) {
        if (capacity > _capacity - _usage[_index]) {
            if (_index + 1 >= _start.size()) {
                std::cout << "Allocate new memory!" << std::endl;
                AlnScoreType* start;
//                start = static_cast<AlnScoreType*>(std::malloc(_capacity * sizeof(AlnScoreType)));
                start = (AlnScoreType*)_mm_malloc(_capacity * sizeof(AlnScoreType), 32);
                if (!start) {
                    throw std::bad_alloc(); // Handle allocation failure
                }
                _start.push_back(start);
                _usage.push_back(0);
            }
            _index++;
        }

        AlnScoreType* data = _start[_index] + _usage[_index];
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
    std::vector<AlnScoreType*> _start;
    std::vector<size_t> _usage;
    size_t _capacity;
};
