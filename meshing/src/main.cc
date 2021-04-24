
#include "las/read_las.h"
#include "recon/dt.h"
#include "geodset.h"

#include <opencv2/highgui.hpp>

int main() {

  std::string dir = "/data/pointCloudStuff/pc/laplata";

  MultiLasDataset mld({dir});
  mld.split();

  auto tile0 = mld.getTile(269, 3492);
  tile0.load();

  std::sort(tile0.pts.begin(), tile0.pts.end(), [](
        const Eigen::Vector3f& a,
        const Eigen::Vector3f& b) { return a(0) < b(0); });

  /*
  std::stable_sort(tile0.pts.begin(), tile0.pts.end(), [](
        const Eigen::Vector3f& a,
        const Eigen::Vector3f& b) {
        return static_cast<int>(a(0)/50.f) < static_cast<int>(b(0)/50.f);
      });
      */

  SimpleGeoDataset dset("/data/pointCloudStuff/img/laplata/laplata.utm3.tif");
  cv::Mat img;
  dset.tlbrNative(tile0.tlbr, 1024,1024, img);
  cv::imshow("raster",img);
  cv::waitKey(0);

  DelaunayTetrahedrialization dt({false});

  Eigen::Map<RowMatrixX> pts((float*)tile0.pts.data(), tile0.pts.size(), 3);
  dt.run(pts);


  return 0;
}
