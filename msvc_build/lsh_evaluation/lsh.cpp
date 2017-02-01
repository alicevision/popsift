#include "lsh.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <set>

std::mt19937_64 LSHFunction::Engine;

LSHFunction::LSHFunction(unsigned k) :
    _distribution(0, 32767)
{
    // RNG may return duplicates, so use set to generate exactly k different bits.
    // We also need coordinates to be sorted.
    std::set<unsigned short> coordinates;
    while (coordinates.size() < k)
        coordinates.insert(_distribution(Engine));
    _coordinates.insert(_coordinates.end(), coordinates.begin(), coordinates.end());
    if (_coordinates.size() != k)
        throw std::logic_error("LSHFunction: cardinality < k");
}

