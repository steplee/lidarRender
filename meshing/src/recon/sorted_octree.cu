#include "sorted_octree.h"
#include <torch/torch.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "helper_cuda.hpp"

#include <time.h>
inline int64_t get_time_micros() {
  timespec start;
  clock_gettime(CLOCK_REALTIME, &start);
  return start.tv_sec * 1'000'000 + start.tv_nsec / 1'000;
}

static void deleter(void *a){}
// Worlds worst bin search...
// Tested in testSearch.cc
#if 0
__device__ int64_t binSearch(int x, int y, int z, const int64_t* inds, int N) {
  int lo = 0;
  int hi = N-1;
  int steps=0;

  // Search X
  while (lo!=hi) {
    if (inds[lo] < x) lo = (lo+N)/2;
    else if (inds[lo] > x) lo = (lo) / 2;
    else break;
    if(lo == 0) break;
    steps++; if (steps>100) printf(" - STUCK look %d %d %d at X %d %d\n", x,y,z, lo, hi);
    if (lo<0 or hi>N) printf(" - OUTOFBOUNDS look %d %d %d at X1 %d %d %d\n", x,y,z,lo,hi);
  }
  if (inds[lo] != x) return -1;
  while(lo > 0 and inds[lo-1] == x) lo--;
  hi = lo;
  int step = N - lo;
  while (hi < N-1 and inds[hi+1] == x) {
    if (inds[hi+step] == x) hi += step;
    step = step/2 | 1;
    steps++; if (steps>100) printf(" - STUCK look %d %d %d at X2 %d %d\n", x,y,z, lo, hi);
    if (lo<0 or hi>N) printf(" - OUTOFBOUNDS look %d %d %d at X2 %d %d %d\n", x,y,z,lo,hi);
  }
  int lastHi = hi;
  int lastLo = lo;

  // Search Y
  while (lo!=hi) {
    if (inds[lo+N] < y) lo = (lo+lastHi+1)/2;
    else if (inds[lo+N] > y) lo = (lo+lastLo)/2;
    else break;
    steps++; if (steps>100) printf(" - STUCK look %d %d %d at Y %d %d\n", x,y,z, lo, hi);
    if (lo<0 or hi>N) printf(" - OUTOFBOUNDS look %d %d %d at Y1 %d %d %d\n", x,y,z,lo,hi);
  }
  if (inds[lo+N] != y) return -1;
  while(lo > lastLo and inds[lo-1+N] == y) lo--;
  hi = lo;
  step = lastHi+1 - lo;
  while (hi < lastHi+1 and hi < N-1 and inds[hi+1+N] == y) {
    if (inds[hi+step+N] == y) hi += step;
    step = step/2 | 1;
    if (lo<0 or hi>N) printf(" - OUTOFBOUNDS look %d %d %d at Y2 %d %d %d\n", x,y,z,lo,hi);
    steps++; if (steps>100) printf(" - STUCK look %d %d %d at Y2 %d %d\n", x,y,z, lo, hi);
  }
  lastHi = hi;
  lastLo = lo;

  // Search Z
  while (lo!=hi) {
    if (inds[lo+N+N] < z) lo = (lo+lastHi+1)/2;
    else if (inds[lo+N+N] > z) lo = (lo+lastLo) / 2;
    else break;
    if (lo<0 or hi>N) printf(" - OUTOFBOUNDS look %d %d %d at Z1 %d %d %d\n", x,y,z,lo,hi);
    steps++; if (steps>100) printf(" - STUCK look %d %d %d at Z %d %d\n", x,y,z, lo, hi);
  }
  if (inds[lo+N+N] != z) return -1;
  while(lo > lastLo and inds[lo-1+N+N] == z) lo--;
  hi = lo;
  step = lastHi+1 - lo;
  while (hi < lastHi+1 and hi < N-1 and inds[hi+1+N+N] == z) {
    if (inds[hi+step+N+N] == z) hi += step;
    step = step/2 | 1;
    if (lo<0 or hi>N) printf(" - OUTOFBOUNDS look %d %d %d at Z2 %d %d %d\n", x,y,z,lo,hi);
    steps++; if (steps>100) printf(" - STUCK look %d %d %d at Z2 %d %d\n", x,y,z, lo, hi);
  }

  return hi == lo ? lo : -1;
}
#else
__device__ int64_t static binSearch(int x, int y, int z, const int64_t* inds, int N) {

  // Naive O(n) kernel
  //for (int i=0; i<N; i++) if (inds[i] == x and inds[i+N] == y and inds[i+N+N] == z) return i;
  //return -1;

  // Binary Search O(n) kernel, but about 100x faster than above (~O(logn))
  // It is O(n) in the worst case because lo/hi step one at a time after 'mid' is found.
  // This is an easy fix by just binary searching the best step for lo/hi,
  // making it O(log(log(n)))
  // Can also get rid of a bunch of additions by adding N to lo/hi for each after X,Y

  int lastLo = 0, lastHi = N;
  int lo=0, hi=N;
  int mid = (lo + hi) / 2;

  // X
  while (lo < hi and inds[mid] != x) {
    if (inds[mid] > x) { hi = mid; mid = (lo+hi) / 2; }
    else if (inds[mid] < x) { lo = mid+1; mid = (lo+hi) / 2; }
  }
  //lo=hi=mid;
  if (inds[lo] != x) lo = mid;
  if (inds[hi] != x) hi = mid;
  while (lo>lastLo+1 and inds[lo-1] == x) lo--;
  while (hi<lastHi   and inds[hi] == x) hi++;
  if (inds[lo] != x) return -1;
  lastLo = lo, lastHi = hi;
  mid = (lo+hi) / 2;

  // Y
  while (lo < hi and inds[N+mid] != y) {
    if (inds[N+mid] > y) { hi = mid; mid = (lo+hi) / 2; }
    else if (inds[N+mid] < y) { lo = mid+1; mid = (lo+hi) / 2; }
  }
  //lo=hi=mid;
  if (inds[N+lo] != y) lo = mid;
  if (inds[N+hi] != y) hi = mid;
  while (lo>lastLo+1 and inds[N+lo-1] == y) lo--;
  while (hi<lastHi   and inds[N+hi] == y) hi++;
  if (inds[N+lo] != y) return -1;
  lastLo = lo, lastHi = hi;
  mid = (lo+hi) / 2;

  // Z
  while (lo < hi and inds[N+N+mid] != z) {
    if (inds[N+N+mid] > z) { hi = mid; mid = (lo+hi) / 2; }
    else if (inds[N+N+mid] < z) { lo = mid+1; mid = (lo+hi) / 2; }
  }
  //lo=hi=mid;
  if (inds[N+N+lo] != z) lo = mid;
  if (inds[N+N+hi] != z) hi = mid;
  while (lo>lastLo+1 and inds[N+N+lo-1] == z) lo--;
  while (hi<lastHi   and inds[N+N+hi] == z) hi++;
  if (inds[N+N+lo] != z) return -1;

  // We know that after coalesce() indices are unique, so either 1 or 0 correct indices.
  return hi==lo+1 ? lo : -1;
}
#endif

// This assumes a block size of 27 and does a bin search for each indepdently.
// A more efficient impl would be to implement the binary search in the kernel,
// and walk the lo/hi bounds for each of XYZ in a single thread, and get parallelsim elsewhere.


// Higher COV_M = Higher quality, larger search radius, but much higher computation.
// 10 Is the max value (11^3 > 1024)
// Must balance to find 
//static constexpr int COV_M = 3;
//static constexpr int COV_M = 9;
static constexpr int COV_M = 5;
static constexpr int COV_N = COV_M*COV_M*COV_M;
__global__ void gatherCovariances_singleLvl(
    float* dst, float* sum,
    const int64_t* inds, const float* vals, int N) {
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
    dst[gid*9 + 0*3 + 0] = dst_[0];
    dst[gid*9 + 1*3 + 1] = dst_[1];
    dst[gid*9 + 2*3 + 2] = dst_[2];
    dst[gid*9 + 1*3 + 0] = dst[gid*9 + 0*3 + 1] = dst_[3];
    dst[gid*9 + 2*3 + 0] = dst[gid*9 + 0*3 + 2] = dst_[4];
    dst[gid*9 + 2*3 + 1] = dst[gid*9 + 1*3 + 2] = dst_[5];
  }

  if (lid == 0 and gid % 40000 == 0) {
    printf(" - cov %d : %f %f %f | %f %f %f | sum %f.\n", gid, dst_[0],dst_[1],dst_[2],dst_[3],dst_[4],dst_[5],sum_[0]);
    //printf("%f %f %f\n%f %f %f\n%f %f %f\n", dst[gid*9+0],dst[gid*9+1],dst[gid*9+2], dst[gid*9+3],dst[gid*9+4],dst[gid*9+5], dst[gid*9+6],dst[gid*9+7],dst[gid*9+8]);
  }
}

// https://docs.nvidia.com/cuda/cusolver/index.html#batchgesvdj-example1

torch::Tensor computeNormals(SortedOctree& so, int lvl) {
  torch::Tensor normals;

  // Form covariance matrix C for each cell.
  // Run SVD/Eig on each C.
  // Extract smallest singular-vector/eigenvector

  auto inds = so.getLvl(lvl).indices();
  auto vals = so.getLvl(lvl).values();
  const int N = inds.size(1);

  float *cov=0, *sum=0;
  checkCudaErrors(cudaMalloc(&cov, N*sizeof(float)*9));
  checkCudaErrors(cudaMalloc(&sum, N*sizeof(float)*1));
  cudaMemset(cov, 0, N*sizeof(float)*9);
  cudaMemset(sum, 0, N*sizeof(float)*1);
  dim3 blk(N), thr(COV_N);
  getLastCudaError("before gather");
  std::cout << " - allocated " << N << "*9 floats at " << cov << "\n";
  printf(" - gatherCovariances: launching threads (%d) (%d).\n", blk.x, thr.x); fflush(stdout);
  cudaDeviceSynchronize();
  auto t0 = get_time_micros();
  gatherCovariances_singleLvl<<<blk,thr>>>(cov, sum, inds.data_ptr<int64_t>(), vals.data_ptr<float>(), N);
  cudaDeviceSynchronize();
  auto t1 = get_time_micros();
  std::cout << " - gather took : " << static_cast<double>(t1-t0) / 1000. << "ms.\n";
  getLastCudaError("after gather");
  std::cout << " - done gather.\n";

  {
#if 1
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

    int info[batchSize];       /* info = [info0 ; info1] */
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

    /* step 1: create cusolver handle, bind a stream  */
    status = cusolverDnCreate(&cusolverH);
    getLastCudaError("after gather");
    assert(CUSOLVER_STATUS_NOT_INITIALIZED != status);
    assert(CUSOLVER_STATUS_ALLOC_FAILED != status);
    assert(CUSOLVER_STATUS_ARCH_MISMATCH != status);
    assert(CUSOLVER_STATUS_SUCCESS == status);

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
    status = cusolverDnXgesvdjSetMaxSweeps( gesvdj_params, 200);
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

    auto infoMult = torch::from_blob((void*)info, {N,1}, deleter, torch::kInt32).cuda().to(torch::kFloat32);
    normals = normals * (infoMult == 0);

    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_work);
    cudaFree(d_S);
#elif 1

    cudaDeviceSynchronize();
    torch::Tensor cov_t = torch::from_blob((void*)cov, {N,3,3}, deleter, torch::kCUDA);
    cudaDeviceSynchronize();
    //std::cout << " - cov_t : " << cov << " " << cov_t.sizes() << cov_t.dtype() << cov_t.device() << std::endl;
    //std::cout << " - cov_t :\n" << cov_t << std::endl;

    auto USV = cov_t.svd(false,true);
    //std::cout << " - U : " << std::get<0>(USV).sizes() << std::endl;
    using namespace torch::indexing;
    normals = std::get<0>(USV).index({Slice(), Slice(), -1});
#endif
  }

  return normals;
}




__global__ void assignSubLevel(float* out, const int64_t* inds0, int N0,
                         const float* in,  const int64_t* inds, const float* sum, int N,
                         int lvl) {

  int gid = threadIdx.x + blockIdx.x * blockDim.x;

  int x = inds0[gid], y = inds0[gid+N0], z = inds0[gid+N0*2];

  int sx = x >> lvl, sy = y >> lvl, sz = z >> lvl;

  auto sid = binSearch(sx, sy, sz, inds, N);
  if (sid >= 0 and sid < N) {
    //if (gid % 12260 == 0) printf(" - sub %d %d %d (>>%d = %d %d %d) -> %ld %ld %ld (sum %f)\n", x,y,z,lvl,sx,sy,sz, inds[sid],inds[sid+N],inds[sid+N+N], sum[sid]);


    if (sum[sid] > 4) {
      const float weight = 1. / (1<<lvl);
      out[gid*3+0] += weight * in[sid*3+0];
      out[gid*3+1] += weight * in[sid*3+1];
      out[gid*3+2] += weight * in[sid*3+2];
    }
  }
}

torch::Tensor computeNormals2(SortedOctree& so, int topLvls) {
  torch::Tensor normals;

  // Form covariance matrix C for each cell.
  // Run SVD/Eig on each C.
  // Extract smallest singular-vector/eigenvector

  // Lvl 0 is our base. The output n normals will be the n points in lvl 0.

  auto inds0 = so.getLvl(0).indices();
  auto vals0 = so.getLvl(0).values();
  const int N0 = inds0.size(1);

  torch::Tensor outNormals = torch::zeros({N0,3}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

  float *cov=0, *sum=0;
  checkCudaErrors(cudaMalloc(&cov, N0*sizeof(float)*9));
  checkCudaErrors(cudaMalloc(&sum, N0*sizeof(float)*1));


    const int batchSize0 = N0;
    printf(" - initializing cuSolver (base size %d).\n",batchSize0); fflush(stdout);
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
    const int minmn = (m < n) ? m : n;

    int *info = (int*)malloc(sizeof(int)*batchSize0);       /* info = [info0 ; info1] */
    float *d_A  = cov; /* lda-by-n-by-batchSize0 */
    float *d_U  = NULL; /* ldu-by-m-by-batchSize0 */
    float *d_V  = NULL; /* ldv-by-n-by-batchSize0 */
    float *d_S  = NULL; /* minmn-by-batchSizee */
    int* d_info  = NULL; /* batchSize0 */
    int lwork = 0;       /* size of workspace */
    float *d_work = NULL;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute singular vectors */

    cudaStat2 = cudaMalloc ((void**)&d_U   , sizeof(float)*ldu*m*batchSize0);
    assert(cudaSuccess == cudaStat2);
    cudaStat3 = cudaMalloc ((void**)&d_V   , sizeof(float)*ldv*n*batchSize0);
    assert(cudaSuccess == cudaStat3);
    cudaStat4 = cudaMalloc ((void**)&d_S   , sizeof(float)*minmn*batchSize0);
    assert(cudaSuccess == cudaStat4);
    cudaStat5 = cudaMalloc ((void**)&d_info, sizeof(int   )*batchSize0);
    assert(cudaSuccess == cudaStat5);

    /* step 1: create cusolver handle, bind a stream  */
    status = cusolverDnCreate(&cusolverH);
    getLastCudaError("after gather");
    assert(CUSOLVER_STATUS_NOT_INITIALIZED != status);
    assert(CUSOLVER_STATUS_ALLOC_FAILED != status);
    assert(CUSOLVER_STATUS_ARCH_MISMATCH != status);
    assert(CUSOLVER_STATUS_SUCCESS == status);

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
    status = cusolverDnXgesvdjSetMaxSweeps( gesvdj_params, 200);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* enable sorting */
    status = cusolverDnXgesvdjSetSortEig( gesvdj_params, true);
    assert(CUSOLVER_STATUS_SUCCESS == status);

  // The number of lvls we should use depends on base resolution and dataset spacing.
  for (int lvl=0; lvl<topLvls; lvl++) {
    std::cout << " - Estimating normals on lvl " << lvl << "\n";

    auto inds = so.getLvl(lvl).indices();
    auto vals = so.getLvl(lvl).values();
    const int N = inds.size(1);
    const int batchSize = N;

    cudaMemset(cov, 0, N*sizeof(float)*9);
    cudaMemset(sum, 0, N*sizeof(float)*1);
    dim3 blk(N), thr(COV_N);
    printf(" - gatherCovariances: launching threads (%d) (%d).\n", blk.x, thr.x); fflush(stdout);
    getLastCudaError("before gather");
    cudaDeviceSynchronize();
    auto t0 = get_time_micros();
    gatherCovariances_singleLvl<<<blk,thr>>>(cov, sum, inds.data_ptr<int64_t>(), vals.data_ptr<float>(), N);
    cudaDeviceSynchronize();
    getLastCudaError("after gather");
    std::cout << " - gather took : " << static_cast<double>(get_time_micros()-t0) / 1000. << "ms.\n";

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

    using namespace torch::indexing;
    normals = torch::from_blob((void*)d_U, {N,3,3}, deleter, torch::kCUDA)
      .index({Slice(), -1}).clone();

    auto sumt = torch::from_blob((void*)sum, {N,1}, deleter, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto infoMult = torch::from_blob((void*)info, {N,1}, deleter, torch::kInt32).cuda().to(torch::kFloat32);
    // NaN * 0 is still NaN, so masked fill is safer
    //normals = normals.mul_(infoMult == 0).mul_(sumt > 4);
    normals.masked_fill_(infoMult != 0, 0).masked_fill_(sumt <= 4, 0);

    if (lvl == 0)
      outNormals.copy_(normals);
    else {
      dim3 blk((N0+255)/256), thr(256);
      assignSubLevel<<<blk,thr>>>(
          outNormals.data_ptr<float>(), inds0.data_ptr<int64_t>(), N0,
          normals.data_ptr<float>(), inds.data_ptr<int64_t>(), sum, N, lvl);
      getLastCudaError("post assign");
    }
    std::cout << " - After lvl " << lvl << " have normals:\n" << outNormals.index({Slice(0,-1,48260)}) << "\n";

  }
  cudaFree(d_U);
  cudaFree(d_V);
  cudaFree(d_work);
  cudaFree(d_S);
  cudaFree(cov);
  cudaFree(sum);
  free(info);

  outNormals.divide_(outNormals.norm(2, 1, true));
  return outNormals;
}

