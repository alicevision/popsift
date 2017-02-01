#pragma once
#include <vector>
#include <random>
#include <array>

using Descirptor = std::array<unsigned char, 128>;

class LSHFunction
{
    static std::mt19937_64 Engine;  // Common for all instances to ensure continuous stream

    size_t _m;
    std::vector<unsigned short> _dproj;
    std::array<size_t, 128> _hash_weights;
    std::vector<unsigned short> _bits;

    void generateBitIndexes();
    void generateCoordinateProjections();
    void generateHashWeights();
    size_t map1(unsigned dim, unsigned val);

public:
    LSHFunction(size_t m, unsigned k);
    size_t operator()(const Descirptor&);
};
