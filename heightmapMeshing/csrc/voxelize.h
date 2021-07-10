#include <torch/extension.h>

//void optimizeSurfaceNet_1_cpu(std::vector<float>& verts, const std::vector<int32_t>& vert2tris, const std::vector<int32_t>& tris);
void optimizeSurfaceNet_1_cpu(torch::Tensor& verts, const torch::Tensor& vert2tris, const torch::Tensor& tris);
//void optimizeSurfaceNet_1_cuda(torch::Tensor& verts, const torch::Tensor vert2tris, const torch::Tensor tris);
void optimizeSurfaceNet_1_cuda(torch::Tensor& verts_, const torch::Tensor vert2tris__, const torch::Tensor tris__);
//void optimizeSurfaceNet_1_cuda(std::vector<float>& verts, const std::vector<int32_t>& vert2tris, const std::vector<int32_t>& tris);

// Actually normals are returned.
std::pair<torch::Tensor,torch::Tensor> estimateSurfaceTangents(const torch::Tensor& pts);

std::pair<torch::Tensor,torch::Tensor> filterOutliers(const torch::Tensor& pts, int baseLvl);
