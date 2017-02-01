#pragma once
#include <vector>
#include <random>
#include <array>

class LSHFunction
{
    static std::mt19937_64 Engine;  // Common for all instances to ensure continuous stream

    std::uniform_int_distribution<int> _distribution;
    std::vector<unsigned short> _dproj;
    std::vector<unsigned short> _bits;

public:
    LSHFunction(unsigned k);
};