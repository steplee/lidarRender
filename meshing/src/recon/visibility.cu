#include "sorted_octree.h"
#include <torch/torch.h>
#include "helper_cuda.hpp"

#include "binSearch.cuh"

static void deleter(void *a){}

// Angle 0: < 0, 0,1>
// Angle 1: <-1,-1,3>
// Angle 2: < 1,-1,3>
// Angle 3: < 1, 1,3>
// Angle 4: <-1, 1,3>
constexpr int Angles = 5;
//constexpr int Parts = 100;
constexpr int Parts = 20;
constexpr int Step = 5;


__global__ static void find_angles(
    uint8_t* dst,
    const int64_t* inds, int N) {
  int lid = threadIdx.x;
  int gid = blockIdx.x;

  __shared__ uint8_t path[Angles*Parts];
  path[lid] = 0xff;

  if (gid >= N) return;

  int angle = lid / Parts;
  int part = lid % Parts;


  int x = inds[gid], y = inds[gid+N], z = inds[gid+N*2];

  int dx=0, dy=0, dz0=part*Step;
  if (angle == 1)      dx = -part, dy = -part;
  else if (angle == 2) dx =  part, dy = -part;
  else if (angle == 3) dx =  part, dy =  part;
  else if (angle == 4) dx = -part, dy =  part;

  uint8_t me = 0xff;

  for (int dz=dz0; dz<dz0+Step; dz++) {
    auto nid = binSearch(x+dx, y+dy, z+dz, inds, N);

    if (dz > 0 and nid != -1) me &= (0xff ^ (1<<angle));

    /*
    if (gid == 0) {
      int xx=-1,yy=-1,zz=-1;
      if (nid>=0 and nid<N)
        xx = inds[nid], yy = inds[nid+N], zz = inds[nid+N*2];
      printf(" - path at %d %d %d delta %d %d %d : nid %d at %d %d %d : me %x.\n", x,y,z,dx,dy,dz, nid,xx,yy,zz, me);
    }
    */
  }

  path[lid] = me;

  __syncthreads();
  for (int d=Angles*Parts/2; d>0; d>>=1) {
    if (lid < d and lid+d < Angles*Parts) {
      path[lid] &= path[lid+d];
      __syncthreads();
    }
  }

  if (lid == 0) {
    dst[gid] = path[0];
  }

  /*
  if (lid == 0 and gid % 4001 == 0) {
    printf(" - path %d %d %d: %d%d%d%d%d%d%d%d.\n", gid, angle, part,
        (int)((path[0]&(1<<7)) > 0),
        (int)((path[0]&(1<<6)) > 0),
        (int)((path[0]&(1<<5)) > 0),
        (int)((path[0]&(1<<4)) > 0),
        (int)((path[0]&(1<<3)) > 0),
        (int)((path[0]&(1<<2)) > 0),
        (int)((path[0]&(1<<1)) > 0),
        (int)((path[0]&(1<<0)) > 0));
  }
  */
}


__global__ void bitwiseAnd_SubLevel(
    uint8_t* out, const int64_t* inds0, int N0,
    const uint8_t* in,  const int64_t* inds, int N, int lvl) {

  int gid = threadIdx.x + blockIdx.x * blockDim.x;

  int x = inds0[gid], y = inds0[gid+N0], z = inds0[gid+N0*2];

  int sx = x >> lvl, sy = y >> lvl, sz = z >> lvl;

  auto sid = binSearch(sx, sy, sz, inds, N);
  if (sid >= 0 and sid < N) {
    //if (gid % 42260 == 0) printf(" - sub %d %d %d (>>%d = %d %d %d) -> %ld %ld %ld (mask %d/%d)\n", x,y,z,lvl,sx,sy,sz, inds[sid],inds[sid+N],inds[sid+N+N], out[gid],in[sid]);

    out[gid] = out[gid] & in[sid];
  }
}

torch::Tensor computeVisibilityMasks1(SortedOctree& so, int topLvls) {

  auto inds0 = so.getLvl(0).indices();
  auto vals0 = so.getLvl(0).values();
  const int N0 = inds0.size(1);

  torch::Tensor out = torch::full({N0}, 0xff, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kByte));

  torch::Tensor lvlMask = torch::full({N0}, 0xff, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kByte));

  for (int lvl=0; lvl<topLvls; lvl++) {
    std::cout << " - Estimating visibility on lvl " << lvl << "\n";

    auto inds = so.getLvl(lvl).indices();
    auto vals = so.getLvl(lvl).values();
    const int N = inds.size(1);

    dim3 blk(N), thr(Parts*Angles);
    find_angles<<<blk,thr>>>(lvlMask.data_ptr<uint8_t>(), inds.data_ptr<int64_t>(), N);
    cudaDeviceSynchronize();
    getLastCudaError("post find_angles");

    if (lvl == 0) {
      out.copy_(lvlMask);
    } else {
      dim3 blk((N0+255)/256), thr(256);
      bitwiseAnd_SubLevel<<<blk,thr>>>(
          out.data_ptr<uint8_t>(), inds0.data_ptr<int64_t>(), N0,
          lvlMask.data_ptr<uint8_t>(), inds.data_ptr<int64_t>(), N, lvl);
      getLastCudaError("post bitwiseAnd");
    }
  }

  return out;
}


