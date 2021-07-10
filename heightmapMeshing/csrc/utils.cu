#include <torch/extension.h>

#include <cuda_runtime.h>
#include "helper_cuda.hpp"
#include "bin_search.cuh"

using namespace torch;
using namespace torch::indexing;

__global__ void aggregateDensity(
    float* den, int N, const int64_t* inds,
    const float* curDen,
    const int64_t* curInds, int curN, int lvl) {

  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= N) return;

  int x = inds[gid], y = inds[gid+N], z = inds[gid+N*2];

  int sx = x >> lvl, sy = y >> lvl, sz = z >> lvl;

  auto sid = binSearch(sx, sy, sz, curInds, curN);
  if (sid >= 0 and sid < curN) {
    den[gid] += curDen[sid];
  }
}

std::pair<torch::Tensor,torch::Tensor> filterOutliers(const torch::Tensor& pts_, int baseLvl) {
  torch::Tensor out;

  int BASE_LVL = baseLvl;
  int LVLS = 6;
  //torch::Tensor lvls[LVLS];
  int BASE_SIZE = 1 << BASE_LVL;
  int size = 1 << BASE_LVL;

  torch::Tensor pts = pts_.cuda();

  torch::Tensor lo = std::get<0>(pts.min(0));
  torch::Tensor hi = std::get<0>(pts.max(0));
  float hi_ = hi.max().cpu().item().to<float>();
  pts.sub_(lo).div_(hi_-lo+1e-4);

  int N_ = pts_.size(0);

  //lvls[0] = torch::sparse_coo_tensor(pts.mul((float)size).t().to(kInt64), torch::ones({N_}, TensorOptions().device(kCUDA)), {size,size,size}).coalesce();
  torch::Tensor lvl_0 = torch::sparse_coo_tensor(pts.mul((float)size).t().to(kInt64),
      torch::ones({N_}, TensorOptions().device(kCUDA)), {size,size,size}).coalesce();

  auto inds0 = lvl_0.indices();
  auto vals0 = lvl_0.values();
  int N = vals0.size(0);

  //torch::Tensor density = torch::zeros({N}, TensorOptions().device(kCUDA));
  torch::Tensor density = vals0.clone();

  std::cout << " - lvl0 " << " has " << pts.size(0) << " elements.\n";
  for (int i=1; i<LVLS; i++, size>>=1) {
    size >>= 1;
    //lvls[i] = torch::sparse_coo_tensor(pts.mul((float)size).t().to(kInt64), torch::ones({N_}, TensorOptions().device(kCUDA)), {size,size,size}).coalesce();
    //std::cout << " - lvl" << i << " has " << lvls[i].values().size(0) << " elements.\n";

    float weight = 1. / (1<<i);
    auto lvl_i = torch::sparse_coo_tensor(pts.mul((float)size).t().to(kInt64), torch::full({N_}, weight, TensorOptions().device(kCUDA)), {size,size,size}).coalesce();

    dim3 blk((N+255)/256), thr(256);
    aggregateDensity<<<blk,thr>>>(density.data_ptr<float>(), N, inds0.data_ptr<int64_t>(),
                     lvl_i.values().data_ptr<float>(), lvl_i.indices().data_ptr<int64_t>(), lvl_i.indices().size(1), i);
  }

  torch::Tensor outPts = lvl_0.indices().to(kFloat).t().div_((float)BASE_SIZE).mul_(hi_-lo+1e-7).add_(lo);
  //return outPts;
  //std::cout << " - Densities:\n" << density << "\n";
  return std::make_pair(outPts,density);

}
