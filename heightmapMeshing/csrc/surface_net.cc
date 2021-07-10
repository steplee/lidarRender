#include "voxelize.h"

// Note this in in-place.
//void optimizeSurfaceNet_1_cpu(std::vector<float>& verts, const std::vector<int32_t>& vert2tris, const std::vector<int32_t>& tris) {
void optimizeSurfaceNet_1_cpu(torch::Tensor& verts_, const torch::Tensor& vert2tris_, const torch::Tensor& tris_) {
  int V = verts_.size(0) / 1;
  float alpha = .95;
  int iters = 50;

  float* verts = verts_.data_ptr<float>();
  const int* vert2tris = vert2tris_.data_ptr<int>();
  const int* tris = tris_.data_ptr<int>();


  for (int iter=0; iter<iters; iter++) {
    std::cout << " - [SurfaceNet] (iter " << iter << ") optimizing " << V << "verts\n";
    for (int i=0; i<V; i++) {
      if (i % 100000 == 0) std::cout << "    - on " << i << " / " << V << "\n";
      float mu[3] = {0,0,0};
      float cnt = 0;

      for (int t=0; t<12; t++) {
        if (vert2tris[i*12+t] != -1) {
          int ti = vert2tris[i*12+t];
          cnt += 3;
          //std::cout << " - vert " << i << " has tri " << ti << std::endl;
          for (int j=0; j<3; j++)   // Three incident verts
            for (int k=0; k<3; k++) // Three coordinates
              mu[k] += verts[tris[ti*3+j]*3+k];
        }
      }

      //std::cout << " - mu " << mu[0] << " " << mu[1] << " " << mu[2] << " cnt " << cnt << "\n";
      //std::cout << " - " << verts[i*3+0] << " " << verts[i*3+1] << " " << verts[i*3+2] << " -> ";
      verts[i*3+0] = alpha * verts[i*3+0] + (1-alpha) * mu[0] / cnt;
      verts[i*3+1] = alpha * verts[i*3+1] + (1-alpha) * mu[1] / cnt;
      verts[i*3+2] = alpha * verts[i*3+2] + (1-alpha) * mu[2] / cnt;
      //std::cout << " - " << verts[i*3+0] << " " << verts[i*3+1] << " " << verts[i*3+2] << std::endl;
    }
  }
}
