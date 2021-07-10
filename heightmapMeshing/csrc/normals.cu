#include "voxelize.h"
#include "bin_search.cuh"

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "helper_cuda.hpp"

using namespace torch;
using namespace torch::indexing;

void deleter(void* ptr) {}
void run_svds_get_normals(torch::Tensor& normals, float* cov, int N);

static constexpr int COV_M = 5;
static constexpr int COV_N = COV_M*COV_M*COV_M;
__global__ void gatherCovariances_singleLvl(
    float* dst, float* sum,
    const int64_t* inds, const float* vals, int N, float scale) {
  int lid = threadIdx.x;
  int gid = blockIdx.x;

  __shared__ float sum_[COV_N];
  __shared__ float dst_[COV_N*6]; // 3x3 symmetric matrix = 6 DoF
  sum_[lid] = 0;
  for (int i=0;i<6;i++) dst_[lid*6+i] = 0;

  int x = inds[gid], y = inds[gid+N], z = inds[gid+N*2];
  int dx = (lid % COV_M) - COV_M/2;
  int dy = ((lid/COV_M) % COV_M) - COV_M/2;
  int dz = (lid / (COV_M*COV_M)) - COV_M/2;

  auto nid = binSearch(x+dx, y+dy, z+dz, inds, N);

  if (nid != -1 and nid < N) {
    float v = vals[nid];
    sum_[lid] = v;

    float l2; // Square length of direction
    if (dx==0 and dy==0 and dz==0) l2 = 1e8;
    else l2 = (float)(dx*dx+dy*dy+dz*dz);
    // Diag
    dst_[lid*6 + 0] = v * ((float)dx)*((float)dx) / l2;
    dst_[lid*6 + 1] = v * ((float)dy)*((float)dy) / l2;
    dst_[lid*6 + 2] = v * ((float)dz)*((float)dz) / l2;
    // Off-Diag
    dst_[lid*6 + 3] = v * ((float)dx)*((float)dy) / l2;
    dst_[lid*6 + 4] = v * ((float)dx)*((float)dz) / l2;
    dst_[lid*6 + 5] = v * ((float)dy)*((float)dz) / l2;
  }

  __syncthreads();
  for (int d=COV_N/2; d>0; d>>=1) {
    if (lid < d and lid+d < COV_N) {
      for (int i=0; i<6; i++) dst_[lid*6 + i] += dst_[(lid+d)*6 + i];
      sum_[lid] += sum_[lid+d];
      __syncthreads();
    }
  }

  if (lid == 0) {
    float s = sum_[0];
    sum[gid] = s;
    dst[gid*9 + 0*3 + 0] = scale * dst_[0];
    dst[gid*9 + 1*3 + 1] = scale * dst_[1];
    dst[gid*9 + 2*3 + 2] = scale * dst_[2];
    dst[gid*9 + 1*3 + 0] = dst[gid*9 + 0*3 + 1] = scale * dst_[3];
    dst[gid*9 + 2*3 + 0] = dst[gid*9 + 0*3 + 2] = scale * dst_[4];
    dst[gid*9 + 2*3 + 1] = dst[gid*9 + 1*3 + 2] = scale * dst_[5];
  }

  if (lid == 0 and gid % 40000 == 0) {
    printf(" - cov %d : %f %f %f | %f %f %f | sum %f.\n", gid, dst_[0],dst_[1],dst_[2],dst_[3],dst_[4],dst_[5],sum_[0]);
  }
}

__global__ void assignCovSubLevel(float* out, float* sum, const int64_t* inds0, int N0,
                         const float* in,  const int64_t* inds, const float* sumCur, int N,
                         int lvl) {

  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= N0) return;

  int x = inds0[gid], y = inds0[gid+N0], z = inds0[gid+N0*2];

  int sx = x >> lvl, sy = y >> lvl, sz = z >> lvl;

  auto sid = binSearch(sx, sy, sz, inds, N);
  if (sid >= 0 and sid < N) {
    //if (gid % 12260 == 0) printf(" - sub %d %d %d (>>%d = %d %d %d) -> %ld %ld %ld (sum %f)\n", x,y,z,lvl,sx,sy,sz, inds[sid],inds[sid+N],inds[sid+N+N], sum[sid]);
    if (gid % 12260 == 0)
      //printf(" - sub %d %d %d (>>%d = %d %d %d) -> %ld %ld %ld (%f %f %f)\n",
      //x,y,z,lvl,sx,sy,sz, inds[sid],inds[sid+N],inds[sid+N+N], out[gid*9+0], out[gid*9+5], out[gid*9+8]);
      printf(" - sub %d %d %d (>>%d = %d %d %d) -> %ld %ld %ld (%f %f %f %f %f %f %f %f %f)\n", x,y,z,lvl,sx,sy,sz, inds[sid],inds[sid+N],inds[sid+N+N],
          out[gid*9+0], out[gid*9+1], out[gid*9+2],
          out[gid*9+3], out[gid*9+4], out[gid*9+5],
          out[gid*9+6], out[gid*9+7], out[gid*9+8]);
    if (sumCur[sid] > 4) {
      const float weight = 1. / (1<<(2*lvl));
      out[gid*9+0] += weight * in[sid*9+0];
      out[gid*9+1] += weight * in[sid*9+1];
      out[gid*9+2] += weight * in[sid*9+2];
      out[gid*9+3] += weight * in[sid*9+3];
      out[gid*9+4] += weight * in[sid*9+4];
      out[gid*9+5] += weight * in[sid*9+5];
      out[gid*9+6] += weight * in[sid*9+6];
      out[gid*9+7] += weight * in[sid*9+7];
      out[gid*9+8] += weight * in[sid*9+8];
      sum[gid] += sumCur[sid];
    }
  }
}


std::pair<torch::Tensor,torch::Tensor> estimateSurfaceTangents(const torch::Tensor& pts_) {
  int N_ = pts_.size(0);
  //torch::Tensor out = torch::zeros({N,3}, TensorOptions().device(kCUDA));
  torch::Tensor out;

  const int BASE_LVL = 12;
  const int LVLS = 6;
  torch::Tensor lvls[LVLS];
  int BASE_SIZE = 1 << BASE_LVL;
  int size = 1 << BASE_LVL;

  torch::Tensor pts = pts_.cuda();

  torch::Tensor lo = std::get<0>(pts.min(0));
  torch::Tensor hi = std::get<0>(pts.max(0));
  float hi_ = hi.max().cpu().item().to<float>();
  pts.sub_(lo).div_(hi_-lo+1e-7);

  lvls[0] = torch::sparse_coo_tensor(pts.mul((float)size).t().to(kInt64), torch::ones({N_}, TensorOptions().device(kCUDA)), {size,size,size}).coalesce();

  //const torch::Tensor inds0 = (pts*(float)size).to(kInt64).t();
  //const torch::Tensor vals0 = torch::ones({N}, TensorOptions().device(kCUDA));
  auto inds0 = lvls[0].indices();
  auto vals0 = lvls[0].values();
  int N = vals0.size(0);

  size >>= 1;
  std::cout << " - lvl0 " << " has " << pts.size(0) << " elements.\n";
  for (int i=1; i<LVLS; i++, size>>=1) {
    lvls[i] = torch::sparse_coo_tensor(pts.mul((float)size).t().to(kInt64), torch::ones({N_}, TensorOptions().device(kCUDA)), {size,size,size}).coalesce();
    std::cout << " - lvl" << i << " has " << lvls[i].values().size(0) << " elements.\n";
  }

  //std::cout << " - lvl[0]:\n" << lvls[0].indices().t() << "\n";

  // Compute covariances
  float *cov, *sum, *covCur, *sumCur;
  checkCudaErrors(cudaMalloc(&cov, N*sizeof(float)*9));
  checkCudaErrors(cudaMalloc(&covCur, N*sizeof(float)*9));
  checkCudaErrors(cudaMalloc(&sum, N*sizeof(float)*1));
  checkCudaErrors(cudaMalloc(&sumCur, N*sizeof(float)*1));
  cudaMemset(cov, 0, N*sizeof(float)*9);
  cudaMemset(sum, 0, N*sizeof(float)*1);

  for (int i=0; i<LVLS; i++) {
    cudaMemset(covCur, 0, N*sizeof(float)*9);
    cudaMemset(sumCur, 0, N*sizeof(float)*1);
    torch::Tensor inds, vals;
    if (i == 0) { inds = inds0; vals = vals0; }
    else { inds = lvls[i].indices(); vals = lvls[i].values(); }
    //inds = lvls[i].indices(); vals = lvls[i].values();

    int NN = vals.size(0);
    dim3 blk(NN), thr(COV_N);
    dim3 blk2((N+255)/256), thr2(256);
    gatherCovariances_singleLvl<<<blk,thr>>>(covCur, sumCur, inds.data_ptr<int64_t>(), vals.data_ptr<float>(), NN, (1<<i));
    assignCovSubLevel<<<blk2,thr2>>>(cov, sum, inds0.data_ptr<int64_t>(), N,
                                  covCur, inds.data_ptr<int64_t>(), sumCur, NN, i);
    std::cout << " - gather/assigned " << NN << " cells.\n";
    cudaDeviceSynchronize();
  }

  //torch::Tensor cov_t = torch::from_blob(cov, {N,9}, TensorOptions().device(kCUDA));
  //std::cout << " - cov:\n" << cov_t << "\n";

  //std::cout << " - inds0:\n" << inds0.t() << "\n";

  // Compute SVDs
  run_svds_get_normals(out, cov, N);

  cudaFree(cov);
  cudaFree(sum);
  cudaFree(covCur);
  cudaFree(sumCur);

  torch::Tensor outPts = lvls[0].indices().to(kFloat).t().div_((float)BASE_SIZE).mul_(hi_-lo+1e-7).add_(lo);
  return std::make_pair(outPts,out);
}


void run_svds_get_normals(torch::Tensor& normals, float* cov, int N) {

    printf(" - initializing cuSolver.\n"); fflush(stdout);
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    gesvdjInfo_t gesvdj_params = NULL;
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    const int m = 3; /* 1 <= m <= 32 */
    const int n = 3; /* 1 <= n <= 32 */
    const int lda = m; /* lda >= m */
    const int ldu = m; /* ldu >= m */
    const int ldv = n; /* ldv >= n */
    const int batchSize = N;
    const int minmn = (m < n) ? m : n;
    cudaDeviceSynchronize(); printf(" - after init.\n"); fflush(stdout);

    //int info[batchSize];       /* info = [info0 ; info1] */
    int *info;
    info = (int*)malloc(sizeof(int)*batchSize);
    float *d_A  = cov; /* lda-by-n-by-batchSize */
    float *d_U  = NULL; /* ldu-by-m-by-batchSize */
    float *d_V  = NULL; /* ldv-by-n-by-batchSize */
    float *d_S  = NULL; /* minmn-by-batchSizee */
    int* d_info  = NULL; /* batchSize */
    int lwork = 0;       /* size of workspace */
    float *d_work = NULL;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute singular vectors */

    cudaStat2 = cudaMalloc ((void**)&d_U   , sizeof(float)*ldu*m*batchSize);
    cudaStat3 = cudaMalloc ((void**)&d_V   , sizeof(float)*ldv*n*batchSize);
    cudaStat4 = cudaMalloc ((void**)&d_S   , sizeof(float)*minmn*batchSize);
    cudaStat5 = cudaMalloc ((void**)&d_info, sizeof(int   )*batchSize);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);
    cudaDeviceSynchronize(); printf(" - after memsets.\n"); fflush(stdout);

    /* step 1: create cusolver handle, bind a stream  */
    status = cusolverDnCreate(&cusolverH);
    getLastCudaError("after gather");
    assert(CUSOLVER_STATUS_NOT_INITIALIZED != status);
    assert(CUSOLVER_STATUS_ALLOC_FAILED != status);
    assert(CUSOLVER_STATUS_ARCH_MISMATCH != status);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    cudaDeviceSynchronize(); printf(" - after create solver.\n"); fflush(stdout);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* step 2: configuration of gesvdj */
    status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* default value of tolerance is machine zero */
    status = cusolverDnXgesvdjSetTolerance( gesvdj_params, 1e-7);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* default value of max. sweeps is 100 */
    status = cusolverDnXgesvdjSetMaxSweeps( gesvdj_params, 100);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* enable sorting */
    status = cusolverDnXgesvdjSetSortEig( gesvdj_params, true);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* step 4: query working space of gesvdjBatched */
    status = cusolverDnSgesvdjBatched_bufferSize(
        cusolverH,
        jobz,
        m, n,
        d_A, lda, d_S, d_U, ldu, d_V, ldv,
        &lwork,
        gesvdj_params, batchSize
        );
    assert(CUSOLVER_STATUS_SUCCESS == status);
    cudaDeviceSynchronize(); printf(" - after setup.\n"); fflush(stdout);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    assert(cudaSuccess == cudaStat1);

    /* step 5: compute singular values of A0 and A1 */
    printf(" - running SVDs.\n"); fflush(stdout);
    status = cusolverDnSgesvdjBatched(
        cusolverH,
        jobz,
        m, n,
        d_A, lda, d_S, d_U, ldu, d_V, ldv,
        d_work, lwork, d_info,
        gesvdj_params, batchSize
        );
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    int ngood=0, nbad=0, nbadparam=0;
    for(int i = 0 ; i < batchSize ; i++){
      if ( 0 == info[i] ) ngood++;
      else if ( 0 > info[i] ) nbadparam++;
      else nbad++;
      //if (0>info[i]) std::cout << " - bad param " << i << " " << info[i] << "\n";
    }
    std::cout << " - Status: " << ngood << " / " << nbadparam << " / " << nbad << "\n";

    //torch::Tensor U = torch::from_blob((void*)d_U, {N,3,3}, deleter, torch::kCUDA);
    //torch::Tensor V = torch::from_blob((void*)d_V, {N,3,3}, deleter, torch::kCUDA);
    //torch::Tensor S = torch::from_blob((void*)d_S, {N,3}, deleter, torch::kCUDA);
    //std::cout << " svdU:\n" << U << "\n";
    //std::cout << " svdS:\n" << S << "\n";
    //std::cout << " svdV:\n" << V << "\n";

    using namespace torch::indexing;

    normals = torch::from_blob((void*)d_U, {N,3,3}, deleter, torch::kCUDA)
      //.index({Slice(), Slice(), -1}).clone();
      .index({Slice(), -1}).clone();

    //auto infoMult = torch::from_blob((void*)info, {N,1}, deleter, torch::kInt32).cuda().to(torch::kFloat32);
    //normals = normals * (infoMult == 0);

    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_work);
    cudaFree(d_S);
    free(info);
}
