#pragma once
#include "gpu_buffer.hpp"
#include <vector>

void optimizeSurfaceNet_1_cuda(GpuBuffer<float> &verts_, const GpuBuffer<int32_t> &vert2tris_, const GpuBuffer<int32_t> &tris_);

void makeGeo(
    std::vector<float>& meshVerts,
    std::vector<int>& meshQuads,
    GpuBuffer<float>& outVerts,
    GpuBuffer<int32_t>& outTris,
    GpuBuffer<int4>& outQuads,
    GpuBuffer<int>& outVert2tris,
    const int4* nodes,
    int N,
    int G);
