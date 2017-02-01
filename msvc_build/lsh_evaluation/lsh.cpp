#include "lsh.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <set>

std::mt19937_64 LSHFunction::Engine;

LSHFunction::LSHFunction(size_t m, unsigned k) :
    _m(m), _dproj(129, 0)
{
    _dproj[128] = k;
    _bits.reserve(k+1);

    generateBitIndexes();
    generateCoordinateProjections();
    generateHashWeights();
}

void LSHFunction::generateBitIndexes()
{
    std::uniform_int_distribution<unsigned short> distribution(0, 32767);
    std::set<unsigned short> coordinates;
    const unsigned k = _dproj[128];

    while (coordinates.size() < k)
        coordinates.insert(distribution(Engine));
    _bits.insert(_bits.end(), coordinates.begin(), coordinates.end());
    if (_bits.size() != k)
        throw std::logic_error("LSHFunction: cardinality < k");
    _bits.push_back(32768);
}

void LSHFunction::generateCoordinateProjections()
{
    const unsigned k = _dproj[128];
    
    unsigned i = 0;
    while (i < k) {
        unsigned d = _bits[i] >> 8;
        _dproj[d] = i;
        while (++i < k && (_bits[i] >> 8) == d)
            ;
    }
}

void LSHFunction::generateHashWeights()
{
    std::uniform_int_distribution<size_t> distribution(0, _m-1);
    std::generate(_hash_weights.begin(), _hash_weights.end(), [&]() { return distribution(Engine); });
}

size_t LSHFunction::map1(unsigned dim, unsigned val)
{
    unsigned short* b = &_bits[_dproj[dim]];
    unsigned short* e = &_bits[_dproj[dim + 1]];
    // XXX: TODO
}

size_t LSHFunction::operator()(const Descirptor& desc)
{
    size_t h = 0;
    for (int i = 0; i < 128; ++i)
        h = (h + map1(i, desc[i])) % _m;
}