#include "voxelize.h"

__global__ void optimizeSurafceNet1(
    float* outVerts, const float* inVerts, int V,
    const int* vert2tris,
    const int* tris,
    float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= V) return;

  //int j = threadIdx.y;
  //__shared__ int cnts[12];
  //__shared__ float sums[3*12];
  //cnts[j] = 0;

  int cnt = 0;
  float mu[3] = {0,0,0};

  for (int t=0; t<12; t++) {
    if (vert2tris[i*12+t] != -1) {
      int ti = vert2tris[i*12+t];
      cnt += 3;
      for (int j=0; j<3; j++)   // Three incident verts
        for (int k=0; k<3; k++) // Three coordinates
          mu[k] += inVerts[tris[ti*3+j]*3+k];
    }
  }

  outVerts[i*3+0] = alpha * inVerts[i*3+0] + (1-alpha) * mu[0] / cnt;
  outVerts[i*3+1] = alpha * inVerts[i*3+1] + (1-alpha) * mu[1] / cnt;
  outVerts[i*3+2] = alpha * inVerts[i*3+2] + (1-alpha) * mu[2] / cnt;
}

void optimizeSurfaceNet_1_cuda(torch::Tensor &verts_, const torch::Tensor vert2tris__, const torch::Tensor tris__) {
  std::cout << " - Allocating GPU Buffers:\n";
  std::cout << "        - verts    : " << verts_.nbytes()  / 1024./1024. << "mb\n";
  std::cout << "        - vert2tris: " << vert2tris__.nbytes()  / 1024./1024. << "mb\n";
  std::cout << "        - tris     : " << tris__.nbytes()  / 1024./1024. << "mb\n";
  verts_ = verts_.cuda();
  torch::Tensor verts_2 = verts_.clone();

  torch::Tensor tris_ = tris__.cuda();
  torch::Tensor vert2tris_ = vert2tris__.cuda(); // This is a lot of data, we don't have space for
  //return;

  int V = verts_.size(0) / 1;
  //float alpha = .6; int iters = 5;
  //float alpha = .9; int iters = 20;
  //float alpha = .95; int iters = 50;
  float alpha = .95; int iters = 20;
  //float alpha = .65; int iters = 1;



  float* verts1 = verts_.data_ptr<float>();
  float* verts2 = verts_2.data_ptr<float>();
  const int* vert2tris = vert2tris_.data_ptr<int>();
  const int* tris = tris_.data_ptr<int>();

  dim3 blk((V+255)/256), thr(256);
  for (int ii=0; ii<iters; ii++) {
    if (ii % 2 == 0)
      optimizeSurafceNet1<<<blk,thr>>>(verts1, verts2, V, vert2tris, tris, alpha);
    else
      optimizeSurafceNet1<<<blk,thr>>>(verts2, verts1, V, vert2tris, tris, alpha);
  }

  if (iters % 2 == 0) verts_.copy_(verts_2);

}
