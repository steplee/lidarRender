#pragma once

//#include "octree.h"
template<class V>
class BakedOctree;
class Float4;

struct SOR {
  float omega = 1.333f;
  void run(BakedOctree<Float4>& tree);
};

static __global__ void gauss_seidel_red_black(
    float4* out,
    const float4* in,
    const int32_t* pts, const int32_t* neighbors, int N, bool isEven) {
}

void SOR::run(BakedOctree<Float4>& tree) {
  float4 *vals1 = (float4*) tree.vals;
}
