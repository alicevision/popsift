#pragma once
#include <vector>
#include <random>

class LSHFunction
{
    static std::mt19937_64 Engine;  // Common for all instances to ensure continuous stream

    std::uniform_int_distribution<int> _distribution;
    std::vector<unsigned short> _coordinates;

public:
    LSHFunction(unsigned k);
};