#pragma once

#include <ATen/Tensor.h>
namespace torch { using at::Tensor; };

void agg(const torch::Tensor& a);

constexpr int MAX_LVLS = 20;

struct SortedOctree {
  torch::Tensor lvls[MAX_LVLS];
  inline void setLvl(int i, const torch::Tensor& l) { lvls[i] = l; }
  inline torch::Tensor& getLvl(int i) { return lvls[i]; }
  int residentLvls=0;

  double baseTransform[16];
};
struct CompressedOctree {
  torch::Tensor sp; // Has 4 indices (depth, x,y,z)
  double baseTransform[16];
  int maxLvl;
};

// Compute normals using only one octree level.
torch::Tensor computeNormals(SortedOctree& so, int lvl);
// Compute normals using a few highest res levels.
torch::Tensor computeNormals2(SortedOctree& so, int topLvls);

// TODO
// Compute using few highest res faster.
torch::Tensor computeNormals3(SortedOctree& so);

torch::Tensor computeVisibilityMasks1(SortedOctree& so, int topLvls);
torch::Tensor computeVisibilityMasks2(SortedOctree& so, int topLvls);


void completeOctree(SortedOctree& so);
void compressOctree(CompressedOctree& out, SortedOctree& in);

void getPointsAtNodeCenters(std::vector<float>& pts, SortedOctree& so, int lvl);
void getPointsAtCompressedNodeCenters(std::vector<float>& pts, CompressedOctree& so);
