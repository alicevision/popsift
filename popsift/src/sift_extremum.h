#pragma once

namespace popart {

struct Extremum
{
    float xpos;
    float ypos;
    float sigma; // scale;
    float orientation;
};

struct Descriptor
{
    float features[128];
};

} // namespace popart
