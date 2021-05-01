#include "sorted_octree.h"

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "helper_cuda.hpp"

#include "binSearch.cuh"

using namespace torch::indexing;
using namespace torch;

// It seems to do this you must first check if neighbor on same level exists,
// then only if not, create the parent of the missing neighbor.
// However you don't need to do that since a recursive octree will always
// have parents for all neighbors!
__global__ static void ensure_step(
    int64_t* indOut, float* valOut,
    int lvlC,
    const int64_t* indsP, int np,
    const int64_t* indsC, int nc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > nc) return;

  int x = indsC[i], y = indsC[i+nc], z = indsC[i+nc+nc];
  int lvlP = lvlC/2;

  int dx = (x & 1) * 2 - 1;
  int dy = (y & 1) * 2 - 1;
  int dz = (z & 1) * 2 - 1;
  if (x/2+dx > 0 and x/2+dx < lvlP) {
    int nid = binSearch(x/2+dx,y/2,z/2, indsP,np);
    if (nid == -1) {
      indOut[4*(0*nc + i) +0] = x/2+dx;
      indOut[4*(1*nc + i) +0] = y/2   ;
      indOut[4*(2*nc + i) +0] = z/2   ;
      valOut[4*i +0]          = 1e-9;
    }
  }
  if (y/2+dy > 0 and y/2+dy < lvlP) {
  int nid = binSearch(x/2,y/2+dy,z/2, indsP,np);
  if (nid == -1) {
    indOut[4*(0*nc + i) +1] = x/2   ;
    indOut[4*(1*nc + i) +1] = y/2+dy;
    indOut[4*(2*nc + i) +1] = z/2   ;
    valOut[4*i +1]          = 1e-9;
  }
  }
  if (z/2+dz > 0 and z/2+dy < lvlP) {
  int nid = binSearch(x/2,y/2,z/2+dz, indsP,np);
  if (nid == -1) {
    indOut[4*(0*nc + i) +2] = x/2   ;
    indOut[4*(1*nc + i) +2] = y/2   ;
    indOut[4*(2*nc + i) +2] = z/2+dz;
    valOut[4*i +2]          = 1e-9;
  }
  }

  if (dx == dy and dy == dz
      and x/2+dx>=0 and x/2+dx<lvlP
      and y/2+dy>=0 and y/2+dy<lvlP
      and z/2+dz>=0 and z/2+dz<lvlP) {
    int nid = binSearch(x/2+dx,y/2+dy,z/2+dz, indsP,np);
    if (nid == -1) {
      indOut[4*(0*nc + i) +3] = x/2+dx;
      indOut[4*(1*nc + i) +3] = y/2+dy;
      indOut[4*(2*nc + i) +3] = z/2+dz;
      valOut[4*i +3]          = 1e-9;
    }
  }

}

__global__ static void complete_parents(
    int16_t* indOut, float* valOut,
    const int64_t* indsP, int np,
    const int64_t* indsC, const float* valsC, int nc
    ) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i > np) return;

  int x = indsP[i], y = indsP[i+np], z = indsP[i+np+np];

  bool need = false;
  // Should we only require cells with actual points to be completed?
  // Or also require added cells from ensure_step() to complete?
  // Has 2x-4x more memory for the latter.
#if 0
  int nid;
  nid = binSearch(x*2+0,y*2+0,z*2+0, indsC, nc); if (nid >= 0 and valsC[nid] > .2) need = true;
  nid = binSearch(x*2+1,y*2+0,z*2+0, indsC, nc); if (nid >= 0 and valsC[nid] > .2) need = true;
  nid = binSearch(x*2+1,y*2+1,z*2+0, indsC, nc); if (nid >= 0 and valsC[nid] > .2) need = true;
  nid = binSearch(x*2+0,y*2+1,z*2+0, indsC, nc); if (nid >= 0 and valsC[nid] > .2) need = true;
  nid = binSearch(x*2+0,y*2+0,z*2+1, indsC, nc); if (nid >= 0 and valsC[nid] > .2) need = true;
  nid = binSearch(x*2+1,y*2+0,z*2+1, indsC, nc); if (nid >= 0 and valsC[nid] > .2) need = true;
  nid = binSearch(x*2+1,y*2+1,z*2+1, indsC, nc); if (nid >= 0 and valsC[nid] > .2) need = true;
  nid = binSearch(x*2+0,y*2+1,z*2+1, indsC, nc); if (nid >= 0 and valsC[nid] > .2) need = true;
#else
  need |= binSearch(x*2+0,y*2+0,z*2+0, indsC, nc) != -1;
  need |= binSearch(x*2+1,y*2+0,z*2+0, indsC, nc) != -1;
  need |= binSearch(x*2+1,y*2+1,z*2+0, indsC, nc) != -1;
  need |= binSearch(x*2+0,y*2+1,z*2+0, indsC, nc) != -1;
  need |= binSearch(x*2+0,y*2+0,z*2+1, indsC, nc) != -1;
  need |= binSearch(x*2+1,y*2+0,z*2+1, indsC, nc) != -1;
  need |= binSearch(x*2+1,y*2+1,z*2+1, indsC, nc) != -1;
  need |= binSearch(x*2+0,y*2+1,z*2+1, indsC, nc) != -1;
#endif

  if (need) {
  indOut[8*(0*np + i)  +0] = 2*x  ;
  indOut[8*(0*np + i)  +1] = 2*x+1;
  indOut[8*(0*np + i)  +2] = 2*x+1;
  indOut[8*(0*np + i)  +3] = 2*x  ;
  indOut[8*(0*np + i)  +4] = 2*x  ;
  indOut[8*(0*np + i)  +5] = 2*x+1;
  indOut[8*(0*np + i)  +6] = 2*x+1;
  indOut[8*(0*np + i)  +7] = 2*x  ;

  indOut[8*(1*np + i)  +0] = 2*y  ;
  indOut[8*(1*np + i)  +1] = 2*y  ;
  indOut[8*(1*np + i)  +2] = 2*y+1;
  indOut[8*(1*np + i)  +3] = 2*y+1;
  indOut[8*(1*np + i)  +4] = 2*y  ;
  indOut[8*(1*np + i)  +5] = 2*y  ;
  indOut[8*(1*np + i)  +6] = 2*y+1;
  indOut[8*(1*np + i)  +7] = 2*y+1;

  indOut[8*(2*np + i)  +0] = 2*z  ;
  indOut[8*(2*np + i)  +1] = 2*z  ;
  indOut[8*(2*np + i)  +2] = 2*z  ;
  indOut[8*(2*np + i)  +3] = 2*z  ;
  indOut[8*(2*np + i)  +4] = 2*z+1;
  indOut[8*(2*np + i)  +5] = 2*z+1;
  indOut[8*(2*np + i)  +6] = 2*z+1;
  indOut[8*(2*np + i)  +7] = 2*z+1;

  valOut[8*i+0] =
  valOut[8*i+1] =
  valOut[8*i+2] =
  valOut[8*i+3] =
  valOut[8*i+4] =
  valOut[8*i+5] =
  valOut[8*i+6] =
  valOut[8*i+7] = 1e-9;
  }
}

void completeOctree(SortedOctree& so) {
  const auto cudaLong = TensorOptions().dtype(kLong).device(kCUDA);
  const auto cudaShort = TensorOptions().dtype(kShort).device(kCUDA);
  const auto cudaFloat = TensorOptions().dtype(kFloat).device(kCUDA);

  int sizeC = (1 << (so.residentLvls-1));

  // 1) Ensure each neighbor step < 2
  for (int lvl=0; lvl<so.residentLvls-2; lvl++) {
    std::cout << " - EnsureStep on lvl " << lvl << "\n";
    const auto& spC      = so.getLvl(lvl);
    const auto& indicesC = spC.indices();
    const auto& valsC    = spC.values();
    auto spP      = so.getLvl(lvl+1);
    const auto& indicesP = spP.indices();
    const auto& valsP    = spP.values();

    std::cout << " - Allocating (8*" << indicesC.size(0) << "*4*" << indicesC.size(1) << ") + (4*4*" <<indicesC.size(1) << ")" << std::endl;
    c10::cuda::CUDACachingAllocator::emptyCache();
    Tensor newIndicesP = torch::zeros({indicesC.size(0),indicesC.size(1)*4}, cudaLong);
    Tensor newValsP = torch::zeros({indicesC.size(1)*4}, cudaFloat);
    c10::cuda::CUDACachingAllocator::emptyCache();

    int nc = indicesC.size(1);
    int np = indicesP.size(1);
    const int64_t* indp = indicesP.data_ptr<int64_t>();
    const int64_t* indc = indicesC.data_ptr<int64_t>();
    int64_t* indn = newIndicesP.data_ptr<int64_t>();
    float* valn = newValsP.data_ptr<float>();

    dim3 blk((nc+511)/512), thr(512);

    ensure_step<<<blk,thr>>>(
        indn,valn,
        sizeC,
        indp, np,
        indc, nc);

    cudaDeviceSynchronize();

    std::cout << " - Selecting indices" << std::endl;
    newIndicesP = newIndicesP.masked_select((newValsP > 0).unsqueeze_(0)).view({3,-1});
    c10::cuda::CUDACachingAllocator::emptyCache();
    std::cout << " - Selecting values" << std::endl;
    newValsP = newValsP.masked_select(newValsP > 0);
    c10::cuda::CUDACachingAllocator::emptyCache();
    std::cout << " - Selected " << newIndicesP.sizes() << " " << newValsP.sizes() << std::endl;
    //std::cout << " - max new index coo:\n" << std::get<0>(newIndicesP.max(1)).t() << "\n";
    //std::cout << " - min new index coo:\n" << std::get<0>(newIndicesP.min(1)).t() << "\n";
    //std::cout << " - max old index coo:\n" << std::get<0>(indicesP.max(1)).t() << "\n";
    //std::cout << " - min old index coo:\n" << std::get<0>(indicesP.min(1)).t() << "\n";

    spP = (spP + torch::sparse_coo_tensor(newIndicesP, newValsP, spP.sizes())).coalesce();
    std::cout << " - old vs new nnz: " << np << " " << spP.indices().size(1) << "\n";
    so.setLvl(lvl+1, spP);
    sizeC >>= 1;
  }

  // 2) Complete each parent
  for (int lvl=so.residentLvls-1; lvl>=1; lvl--) {
    std::cout << " - CompleteParent on lvl " << lvl << "\n";
    auto spP      = so.getLvl(lvl);
    auto indicesP = spP.indices();
    auto valsP    = spP.values();
    auto spC      = so.getLvl(lvl-1);
    auto indicesC = spC.indices();
    auto valsC    = spC.values();
    c10::cuda::CUDACachingAllocator::emptyCache();

    // Whats faster:
    //        A) Creating 9x elements, copying old to :1, filling the last -8: with new values, masked select, coalesce()
    //        B) Creating 8x elements, masked select, add old sparse tensor, coalesce
    // ?
    /*
    Tensor newIndicesC = torch::zeros({indicesC.size(0),indicesC.size(1)*9}, cudaLong);
    Tensor newValsC = torch::zeros({indicesC.size(0),indicesC.size(1)*9}, cudaFloat);
    newIndicesC.index({Slice(), Slice(0,indicesC.size(1))}).copy_(indicesC);
    newValsC.index({Slice(), Slice(0,indicesC.size(1))}).copy_(valsC);
    */
    std::cout << " - Selecting indices" << std::endl;
    Tensor newIndicesC = torch::zeros({indicesC.size(0),indicesP.size(1)*8}, cudaShort);
    c10::cuda::CUDACachingAllocator::emptyCache();
    std::cout << " - Selecting values" << std::endl;
    Tensor newValsC = torch::zeros({indicesP.size(1)*8}, cudaFloat);
    c10::cuda::CUDACachingAllocator::emptyCache();

    int np = indicesP.size(1);
    int nc = indicesC.size(1);
    dim3 blk((np+511)/512), thr(512);
    complete_parents<<<blk,thr>>>(
        newIndicesC.data_ptr<int16_t>(), newValsC.data_ptr<float>(),
        indicesP.data_ptr<int64_t>(), np,
        indicesC.data_ptr<int64_t>(), valsC.data_ptr<float>(), nc
        );

    newIndicesC = newIndicesC.masked_select((newValsC > 0).unsqueeze_(0)).view({3,-1}).to(torch::kLong);
    c10::cuda::CUDACachingAllocator::emptyCache();
    newValsC = newValsC.masked_select(newValsC > 0);
    c10::cuda::CUDACachingAllocator::emptyCache();

    spC = (spC + torch::sparse_coo_tensor(newIndicesC, newValsC, spC.sizes())).coalesce();
    c10::cuda::CUDACachingAllocator::emptyCache();
    std::cout << " - created " << (spC.indices().size(1)-nc) << " new children, " << spC.indices().size(1) << " total.\n";
    so.setLvl(lvl-1, spC);
    c10::cuda::CUDACachingAllocator::emptyCache();
  }

}

void getPointsAtNodeCenters(std::vector<float>& pts, SortedOctree& so, int lvl) {
  const auto& sp = so.getLvl(lvl);
  const auto& inds = sp.indices();
  //const auto& vals = sp.values();

  int m = pts.size();
  int N = inds.size(1);

  pts.resize(m+N*3);

  float lvlScale = 1<<(so.residentLvls-1-lvl);

  torch::Tensor indsCpu = inds.to(torch::kFloat).div_(lvlScale).t().cpu().contiguous();
  //std::cout << " -indsCpu " << indsCpu << "\n";
  memcpy(pts.data()+m, indsCpu.data_ptr<float>(), sizeof(float)*N*3);
}

void getPointsAtCompressedNodeCenters(std::vector<float>& pts, CompressedOctree& so) {
  const auto& sp = so.sp;
  const auto& inds = sp.indices();

  int m = pts.size();
  int N = inds.size(1);

  pts.resize(m+N*3);

  using namespace torch;
  torch::Tensor lvlScales = torch::full({1,N}, 2, TensorOptions().dtype(kFloat).device(kCUDA)).pow_(so.maxLvl-1-inds.index({Slice(0,1)}));

  torch::Tensor indsCpu = inds
    .index({Slice(1,4)})
    .to(torch::kFloat)
    .div_(lvlScales)
    .add_((1.f/lvlScales).div_(2))
    .t().cpu().contiguous();
  memcpy(pts.data()+m, indsCpu.data_ptr<float>(), sizeof(float)*N*3);
}




static __global__ void compressOctreeLvl(
    int64_t* indsOut, float* valsOut,
    int lvl,
    const int64_t* inds, const float* vals, int n,
    const int64_t* indsBelow, int nBelow,
    const int64_t* indsAbove, int nAbove) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > n) return;
  int x = inds[i], y = inds[i+n], z = inds[i+n+n];

  int sub0 = binSearch(x*2,y*2,z*2, indsBelow, nBelow);

  if (sub0 == -1) {
    indsOut[n*0 + i] = lvl;
    indsOut[n*1 + i] = x;
    indsOut[n*2 + i] = y;
    indsOut[n*3 + i] = z;
    valsOut[i] = vals[i];
  } else
    indsOut[n*0 + i] = indsOut[n*1 + i] = indsOut[n*2 + i] = indsOut[n*3 + i] = -1;

}

void compressOctree(CompressedOctree& out, SortedOctree& in) {
  memcpy(out.baseTransform, in.baseTransform, sizeof(out.baseTransform));
  out.maxLvl = in.residentLvls;

  const auto& sp0 = in.getLvl(0);
  const auto& inds0 = sp0.indices();
  int n0 = inds0.size(1);
  torch::Tensor accInds = torch::zeros({4,n0}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
  accInds.index({Slice(1,4)}).copy_(inds0);
  torch::Tensor accVals = sp0.values();

  for (int i=1; i<out.maxLvl-1; i++) {
    const auto& spBelow = in.getLvl(i-1), sp = in.getLvl(i), spAbove = in.getLvl(i+1);
    int64_t* inds = sp.indices().data_ptr<int64_t>(); float *vals = sp.values().data_ptr<float>();
    int64_t* indsBelow = spBelow.indices().data_ptr<int64_t>();
    int64_t* indsAbove = spAbove.indices().data_ptr<int64_t>();
    int ni = sp.indices().size(1);

    torch::Tensor keptInds = torch::zeros({4,ni}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    torch::Tensor keptVals = torch::zeros_like(sp.values());

    dim3 blk((ni+255)/256), thr(256);
    cudaDeviceSynchronize();
    getLastCudaError("pre compressOctreeLvl");
    compressOctreeLvl<<<blk,thr>>>(
        keptInds.data_ptr<int64_t>(), keptVals.data_ptr<float>(), i,
        inds, vals, ni,
        indsBelow, spBelow.indices().size(1),
        indsAbove, spAbove.indices().size(1));
    cudaDeviceSynchronize();
    getLastCudaError("post compressOctreeLvl");


    keptInds = keptInds.masked_select((keptVals > 0).view({1,-1})).view({4,-1});
    keptVals = keptVals.masked_select((keptVals > 0));

    accInds = torch::cat({accInds, keptInds}, 1);
    accVals = torch::cat({accVals, keptVals}, 0);
    std::cout << " - compressed up to lvl " << i << " kept " << keptInds.size(1) << " / " << ni << " have " << accInds.size(1) << " total." << std::endl;
  }

  out.sp = torch::sparse_coo_tensor(accInds, accVals).coalesce();
}
