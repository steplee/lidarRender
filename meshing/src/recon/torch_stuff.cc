#include "torch_stuff.h"
#include "sorted_octree.h"
//#include "visibility.h"

#include <torch/torch.h>

static void buildOctree(
    SortedOctree& oct,
    std::vector<Eigen::Vector3f>& basePts,
    const std::vector<Eigen::Vector3f>& points) {
  using namespace torch::indexing;

  int N = points.size();
  torch::Tensor pts = torch::from_blob((void*)points.data(), {N,3},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).cuda();


  double min[3], max[3];
  auto mm = std::get<0>(pts.min(0)).cpu();
  min[0] = mm.data_ptr<float>()[0];
  min[1] = mm.data_ptr<float>()[1];
  min[2] = mm.data_ptr<float>()[2];
  mm = std::get<0>(pts.max(0)).cpu();
  max[0] = mm.data_ptr<float>()[0];
  max[1] = mm.data_ptr<float>()[1];

  // NOTE XXX TODO: Max 500m alt diff
  // Instead should do quantile + clip
  max[2] = std::min(mm.data_ptr<float>()[2], (float)min[2] + 800.f);
  pts.index({Slice(),2}).clamp_(0, max[2]);

  printf(" - minmax[0]: (%f %f) (size %f).\n", min[0], max[0], max[0]-min[0]);
  printf(" - minmax[1]: (%f %f) (size %f).\n", min[1], max[1], max[1]-min[1]);
  printf(" - minmax[2]: (%f %f) (size %f).\n", min[2], max[2], max[2]-min[2]);

  // 0 1  2  3
  // 4 5  6  7
  // 8 9 10 11
  for (int i=0; i<16; i++) oct.baseTransform[i] = i%4 == i/4;
  double scale = .99999 / std::max(max[0]-min[0], std::max(max[1]-min[1], max[2]-min[2]));
  oct.baseTransform[0] = oct.baseTransform[5] = oct.baseTransform[10] = scale;
  oct.baseTransform[3] = -min[0] * scale;
  oct.baseTransform[7] = -min[1] * scale;
  oct.baseTransform[11] = -min[2] * scale;

  // In-place transform.
  auto T = torch::from_blob(oct.baseTransform, {4,4}, torch::TensorOptions().dtype(torch::kFloat64)).to(torch::kFloat32).cuda();
  torch::matmul_out(pts, pts, T.index({Slice(0,3),Slice(0,3)}).t());
  pts.add_(T.index({Slice(0,3),Slice(3,4)}).t());

  {
    auto mm = std::get<0>(pts.min(0)).cpu();
    min[0] = mm.data_ptr<float>()[0]; min[1] = mm.data_ptr<float>()[1]; min[2] = mm.data_ptr<float>()[2];
    mm = std::get<0>(pts.max(0)).cpu();
    max[0] = mm.data_ptr<float>()[0]; max[1] = mm.data_ptr<float>()[1]; max[2] = mm.data_ptr<float>()[2];
    printf(" - After transform minmax:\n");
    printf(" - minmax[0]: (%f %f) (size %f).\n", min[0], max[0], max[0]-min[0]);
    printf(" - minmax[1]: (%f %f) (size %f).\n", min[1], max[1], max[1]-min[1]);
    printf(" - minmax[2]: (%f %f) (size %f).\n", min[2], max[2], max[2]-min[2]);
  }


  torch::Tensor vals = torch::ones({N}, torch::TensorOptions().device(torch::kCUDA));

  Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor>> baseTransform(oct.baseTransform);
  std::cout << " - Octree BaseTransform:\n" << baseTransform <<"\n";

  //int LVLS = 14;
  //int LVLS = 13;
  int LVLS = 12;
  float expansion0 = (1 << (LVLS-1));
  oct.residentLvls = LVLS;
  int lvlSize = (1 << (LVLS-1));

  pts.mul_(expansion0);
  torch::Tensor sp = torch::sparse_coo_tensor(pts.to(torch::kLong).t(), vals, {lvlSize,lvlSize,lvlSize}).coalesce();

  {
    auto ptsCpu = sp.indices().to(torch::kFloat32).divide_(expansion0).cpu();
    int N = ptsCpu.size(1);
    basePts.resize(N);
    const float* ptsCpu_ = ptsCpu.data_ptr<float>();
    for (int i=0; i<3; i++)
    for (int j=0; j<N; j++) {
      basePts[j](i) = ptsCpu_[i*N+j];
    }
  }


  //std::cout << " - lvl " << LVLS-1 << ": " << sp.sizes() << "\n";
  //oct.setLvl(LVLS-1, sp);
  //for (int i=LVLS-2; i>=0; i--) {
  std::cout << " - Initial npts " << pts.size(0) << "\n";
  std::cout << " - lvl " << 0 << ": " << sp.sizes() << "\tnnz " << sp.indices().size(1) << "\n";
  oct.setLvl(0, sp);
  for (int i=1; i<LVLS; i++) {
    lvlSize >>= 1;
    auto coo = sp.indices().floor_divide(2);
    auto val = sp.values();
    sp = torch::sparse_coo_tensor(coo, val, {lvlSize,lvlSize,lvlSize}).coalesce();
    oct.setLvl(i, sp);
    std::cout << " - lvl " << i << ": " << sp.sizes() << "\tnnz " << sp.indices().size(1) << "\n";
  }
}

void estimateNormals(
    std::vector<Eigen::Vector3f>& normals,
    std::vector<Eigen::Vector3f>& basePts,
    const std::vector<Eigen::Vector3f>& points) {
  using namespace torch::indexing;

  int N = points.size();

  // 1) Move pts to gpu
  // 2) Determine baseTransform
  // 3) Compute octree
  // 4) Estimate normals


  SortedOctree oct;
  buildOctree(oct, basePts, points);

  //auto norms = computeNormals(oct, LVLS-4);
  auto norms = computeNormals2(oct, 7).cpu();
  {
    int N = norms.size(0);
    normals.resize(N);
    const float* norms_ = norms.data_ptr<float>();
    for (int i=0; i<N; i++) {
      normals[i](0) = norms_[i*3+0];
      normals[i](1) = norms_[i*3+1];
      normals[i](2) = norms_[i*3+2];
    }

  }


}

void estimateVisibility(
    SortedOctree& oct,
    std::vector<uint8_t>& visibleAngleMasks,
    std::vector<Eigen::Vector3f>& basePts,
    const std::vector<Eigen::Vector3f>& points, int topLvls) {
  using namespace torch::indexing;
  int N = points.size();

  buildOctree(oct, basePts, points);

  //auto vis = computeVisibilityMasks1(oct, 3).cpu();
  //auto vis = computeVisibilityMasks1(oct, 2).cpu();
  auto vis = computeVisibilityMasks1(oct, topLvls).cpu();
  //std::cout << " - TODO: RE-ENABLE computeVisibilityMasks1.\n";
  //auto vis = computeVisibilityMasks1(oct, 0).cpu();
  uint8_t* vis_ = vis.data_ptr<uint8_t>();

  visibleAngleMasks.resize(vis.size(0));
  for (int i=0; i<visibleAngleMasks.size(); i++) {
    visibleAngleMasks[i] = vis_[i];
  }

}
