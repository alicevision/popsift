#include "lsh.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <set>

std::mt19937_64 LSHFunction::Engine;

LSHFunction::LSHFunction(unsigned k) :
    _distribution(0, 32767),
    _dproj(129, 0)
{
    _dproj[128] = k;
    _bits.reserve(k);

    // RNG may return duplicates, so use set to generate exactly k different bits. We also need coordinates to be sorted.
    {
        std::set<unsigned short> coordinates;
        while (coordinates.size() < k)
            coordinates.insert(_distribution(Engine));
        _bits.insert(_bits.end(), coordinates.begin(), coordinates.end());
        if (_bits.size() != k)
            throw std::logic_error("LSHFunction: cardinality < k");
    }

    {
        unsigned i = 0;
        while (i < k) {
            unsigned d = _bits[i] >> 8;
            _dproj[d] = i;
            while (++i < k && (_bits[i] >> 8) == d)
                ;
        }
    }
}

