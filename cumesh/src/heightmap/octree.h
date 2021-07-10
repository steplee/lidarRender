#pragma once

// Todo: support passing a reduction option (e.g add or avg or max ...)

//struct NoValue {};
struct Float4 {
  float x,y,z,w;
};

template <class T>
struct SortedOctree {
  int64_t* inds = 0;
  //int32_t* cnts = 0;
  T* vals = 0;

  // This will allow starting directly at the correct level, avoiding many bin searches.
  int lvlOffsets[24];
  int lvlSizes[24];

  int maxLvl = -1;
  int minLvl = -1;

  void createOctree(const float* in, const T* vals, int N, int depth, int minLvl);

  void getCoordsValsHost(int32_t* outInds, T* outVals, int lvl);

  void release();
};


// Like the sorted octree, but precompute neighbor's indices so that kernels do
// not need to do binary searches
template <class V>
struct BakedOctree {
  // 4*N sized array of coordinates
  int32_t* pts=0;
  // 27*N sized array of neighbor indices in the tree, relative to beginning of array.
  int32_t* neighbors = 0;
  V* vals = 0;
  int lvlOffsets[24], lvlSizes[24];
  int maxLvl = -1, minLvl = -1;

  void bakeTree(SortedOctree<V>& tree, int minLvl, int maxLvl, bool deleteTree=true);

  void release();
};
