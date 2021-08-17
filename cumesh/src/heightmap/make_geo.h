#pragma once
#include "gpu_buffer.hpp"
#include <vector>

void makeGeo(
    std::vector<float>& meshVerts,
    std::vector<int>& meshQuads,
    GpuBuffer<int3>& outTris,
    GpuBuffer<int4>& outQuads,
    GpuBuffer<int>& outVert2tris,
    const int4* nodes,
    int N,
    int G);
