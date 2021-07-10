#include <vector>
#include "binSearch4.cuh"
#include "helper_cuda.hpp"
#include "octree.h"

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
template <class T>
using DevVector = thrust::device_vector<T>;

namespace thrust {
  template<> Float4 plus<Float4>::operator()(const Float4& a, const Float4& b) const {
    Float4 out;
    out.x = a.x + b.x;
    out.y = a.y + b.y;
    out.z = a.z + b.z;
    out.w = a.w + b.w;
    return out;
  }
};

static __global__ void binPoints_(int3* out, float3* in, int N, int res) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N)
    out[i] = make_int3(in[i].x * res, in[i].y * res, in[i].z * res);
}
static void binPoints(int3* out, const std::vector<float3>& in, int res) {
  float* tmp=0;
  cudaMalloc(&tmp, sizeof(float)*in.size()*3);
  cudaMemcpy(tmp, in.data(), sizeof(float)*in.size()*3, cudaMemcpyHostToDevice);
  binPoints_<<<(in.size()+255)/256,256>>>(out, (float3*)tmp, in.size(), res);
  cudaFree(tmp);
}

static __global__ void binPointsHashed_(int64_t* out, float3* in, int N, int res, int lvl) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    int64_t x = in[i].x * res;
    int64_t y = in[i].y * res;
    int64_t z = in[i].z * res;
    int64_t d = lvl;
    out[i] = (d << 59ul) | (x << 38) | (y << 19) | z;
    //out[i] = 0;
  }
}
static void binPointsHashed(int64_t* out, const float* in, int N, int res, int lvl) {
  binPointsHashed_<<<(N+255)/256,256>>>(out, (float3*)in, N, res, lvl);
}

static __global__ void unhashPoints_(int32_t* out, int64_t* in, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    int64_t pt = in[i];
    out[i*3+0] = decode_x(pt);
    out[i*3+1] = decode_y(pt);
    out[i*3+2] = decode_z(pt);
  }
}
template <class T>
void SortedOctree<T>::getCoordsValsHost(int32_t* outInds, T* outVals, int lvl) {
  int N = lvlSizes[lvl];
  int32_t* tmpInds; cudaMalloc(&tmpInds, sizeof(int32_t)*3*N);
  unhashPoints_<<<(N+256)/256,256>>>(tmpInds, inds+lvlOffsets[lvl], N);
  cudaMemcpy(outInds, tmpInds, sizeof(int32_t)*3*N, cudaMemcpyDeviceToHost);
  cudaFree(tmpInds);
  if (outVals !=0) cudaMemcpy(outVals, vals+lvlOffsets[lvl], sizeof(T)*N, cudaMemcpyDeviceToHost);
}


// To form the sparse tensor of a single level:
//     1) Hash each point to a single 64 bit integer
//     2) Sort them
//     3) thrust::reduce_by_key them (they *must* be sorted)
// Note: if you had some attribute per point, you could use sort_by_key.
// Since I only want to estimate density here, there is no need to sort a constant attribute.
//
// This process is repeated for each level, then all results combined.
// A minor performance increase would recursive use last level, leading to
// less work done over time, but I just use the original pt buffer each lvl.
template <class T>
void SortedOctree<T>::createOctree(
    const float* in, const T* in_vals, int N, int depth, int minLvl) {
  int last_n = N;
  int64_t* lvls[25];

  DevVector<int64_t> out_inds;
  DevVector<T> out_vals;

  if (inds != 0) cudaFree(inds);
  if (vals != 0) cudaFree(vals);

  float* inpts=0;
  cudaMalloc(&inpts, sizeof(float)*N*3);
  cudaMemcpy(inpts, in, sizeof(float)*N*3, cudaMemcpyHostToDevice);
  int64_t* lvlInds=0, *lvlInds2=0;
  T* invals=0, *lvlVals=0, *lvlVals2=0;
  cudaMalloc(&lvlInds, sizeof(int64_t)*N);
  cudaMalloc(&lvlInds2, sizeof(int64_t)*N);

  cudaMalloc(&invals, sizeof(T)*N);
  cudaMemcpy(invals, in_vals, sizeof(T)*N, cudaMemcpyHostToDevice);
  cudaMalloc(&lvlVals, sizeof(T)*N);
  cudaMalloc(&lvlVals2, sizeof(T)*N);

  for (int lvl=0; lvl<24; lvl++)
    lvlSizes[lvl] = lvlOffsets[lvl] = 0;

  for (int lvl=depth; lvl>=minLvl; lvl--) {
    cudaMemcpy(lvlVals, invals, sizeof(T)*N, cudaMemcpyDeviceToDevice);
    int res = 1 << lvl;

    binPointsHashed(lvlInds, inpts, N, res, lvl);
    cudaDeviceSynchronize(); getLastCudaError("post hash");

    thrust::stable_sort_by_key(thrust::device, lvlInds, lvlInds+N, lvlVals);
    //thrust::sort(thrust::device, lvlPts, lvlPts+N);
    cudaDeviceSynchronize(); getLastCudaError("post sort");

    auto itit = thrust::reduce_by_key(thrust::device, lvlInds, lvlInds+N, lvlVals, lvlInds2, lvlVals2);
    int n_lvl = itit.first - lvlInds2;

    int n_prev = out_inds.size();
    lvlOffsets[lvl] = n_prev;
    lvlSizes[lvl] = n_lvl;

    printf(" - [hmr] lvl %2d : %-9d / %-9d (%9d total) (%.2f%% retain).\n", lvl,n_lvl,N, n_prev+n_lvl, 100.f*((float)n_lvl)/last_n);
    out_inds.resize(out_inds.size() + n_lvl);
    out_vals.resize(out_vals.size() + n_lvl);
    thrust::copy(lvlInds2, lvlInds2+n_lvl, out_inds.begin() + n_prev);
    thrust::copy(lvlVals2, lvlVals2+n_lvl, out_vals.begin() + n_prev);

    last_n = n_lvl;
  }

  // Whats up with the device_vector::data function?
  maxLvl = depth;
  minLvl = minLvl;
  if (inpts != 0) cudaFree(inpts);
  if (invals != 0) cudaFree(invals);
  cudaMalloc(&inds, sizeof(int64_t)*out_inds.size());
  cudaMemcpy(inds, thrust::raw_pointer_cast(out_inds.data()), sizeof(int64_t)*out_inds.size(), cudaMemcpyDeviceToDevice);
  cudaMalloc(&vals, sizeof(T)*out_vals.size());
  cudaMemcpy(vals, thrust::raw_pointer_cast(out_vals.data()), sizeof(T)*out_vals.size(), cudaMemcpyDeviceToDevice);

  //cudaFree(pts);
  cudaFree(lvlInds);
  cudaFree(lvlInds2);
  cudaFree(lvlVals);
  cudaFree(lvlVals2);
}


template <class V>
void SortedOctree<V>::release() {
  if (inds) cudaFree(inds);
  if (vals) cudaFree(vals);
  inds = 0; vals = 0;
}


static __global__ void bake_inds_(int32_t* pts, int32_t* neighbors, int64_t* inds, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i > N) return;

  int64_t pt = inds[i];
  int32_t d = decode_d(pt);
  int32_t z = decode_z(pt);
  int32_t y = decode_y(pt);
  int32_t x = decode_x(pt);
  pts[i*4+0] = x;
  pts[i*4+1] = y;
  pts[i*4+2] = z;
  pts[i*4+3] = d;

  for (int dz=-1; dz<=1; dz++)
  for (int dy=-1; dy<=1; dy++)
  for (int dx=-1; dx<=1; dx++) {
    int j = (dz + 1) * 9 + (dy + 1) * 3 + dx+1;
    int off = ((i+27) + j) * 4;
    neighbors[off+0] = binSearch4_encoded(d, x+dx,dy+dy,z+dz, inds, N);
  }

}
template <class V>
void BakedOctree<V>::bakeTree(SortedOctree<V>& tree, int minLvl_, int maxLvl_, bool deleteTree) {
  minLvl = std::max(tree.minLvl, minLvl_);
  maxLvl = std::min(tree.maxLvl, maxLvl_);
  for (int i=0; i<24; i++)
    lvlOffsets[i] = tree.lvlOffsets[i],
    lvlSizes[i] = tree.lvlSizes[i];

  int N = 0;
  for (int i=0; i<24; i++) N += lvlSizes[i];

  cudaMalloc(&pts, sizeof(int32_t)*N*4);
  cudaMalloc(&neighbors, sizeof(int32_t)*N*27);
  if (deleteTree) {
    vals = tree.vals;
    tree.vals = 0;
  } else {
    cudaMalloc(&vals, sizeof(V)*N);
    cudaMemcpy(vals, tree.vals, sizeof(V)*N, cudaMemcpyDeviceToDevice);
  }

  bake_inds_<<<(N+255)/256,256>>>(pts, neighbors, tree.inds, N);

  if (deleteTree) tree.release();
}
template <class V>
void BakedOctree<V>::release() {
  if (pts) cudaFree(pts);
  if (vals) cudaFree(vals);
  if (neighbors) cudaFree(neighbors);
  pts = 0; vals = 0;
  neighbors = 0;
}



template class SortedOctree<int>;
template class SortedOctree<float>;

template class SortedOctree<Float4>;
template class BakedOctree<Float4>;

