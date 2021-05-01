#include "vu.h"

#include "binSearch.cuh"
#include "binSearch4.cuh"

#include "sorted_octree.h"

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "helper_cuda.hpp"

constexpr int Angles = 5;
constexpr int Parts = 20;
constexpr int Step = 5;

__global__ static void walk(
    int16_t *out,
    int64_t* inputInds, int nInputInds,
    int maxLvl,
    const int64_t* inds, int n) {

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > nInputInds) return;
  int i = inputInds[j];

  int d = inds[i], x = inds[i+n], y = inds[i+n+n], z = inds[i+n+n+n];

  // High-level:
  // Walk from the point to the sky.
  // If we hit anything, we count that angle+pt as occluded.
  // Otherwise atomically add to each cell that is walked across.

  // Walking:
  // We first search for the next neighbor along ray on current lvl.
  // This may be absent. In that case we must check both lower/higher depths.
  // If both don't have a neighbor, we must be at an edge of the outer cube.
  //
  // When we enter a new level, we must consider where to exit
  // based on the entrance point and angle.
  //

  const int step = 30;

  for (int i=0; i<30; i++) {
  }

}

void labelCells(CompressedOctree& so, const std::vector<uint8_t>& vis, int lvl) {
  auto& sp = so.sp;
  auto inds = sp.indices();
  auto vals = sp.values();

  int N = inds.size(1);
  assert(N == vis.size());

  torch::Tensor out = torch::zeros({N}, torch::TensorOptions().dtype(torch::kShort).device(torch::kCUDA));

  dim3 blk((N+255)/256), thr(256);
  //walk<<<blk,thr>>>(
}
