#include "read_las.h"

#include <Eigen/Geometry>

#include <iostream>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

#define ENSURE(x)                                                                                  \
    do {                                                                                                 \
        if (!(x)) {                                                                                      \
            std::cout << " - enusre failed " << __LINE__ << " " << __FILE__ << " : " << #x << std::endl; \
            exit(1);                                                                                     \
        }                                                                                                \
    } while (0);


MultiLasDataset::MultiLasDataset(const std::vector<std::string>& dirs)
  : dirs(dirs)
{
  seenOuterTlbr.setZero();
}


void MultiLasDataset::split() {
  for (const auto& dir : dirs) {
      for (auto& p : fs::directory_iterator(dir)) {
        std::string fname = p.path();
        int n = fname.length();

        if ((fname[n-3] == 'l' or fname[n-3] == 'L') and
            (fname[n-2] == 'a' or fname[n-2] == 'A') and
            (fname[n-1] == 'Z' or fname[n-1] == 'S' or fname[n-1] == 'z' or fname[n-1] == 's')) {
          fileNames.push_back(fname);
        }
      }
  }

  ENSURE(fileNames.size() > 0);

  for (int ii=0; ii<fileNames.size(); ii++) {
    const std::string& fname = fileNames[ii];
    Eigen::Vector4f fileTlbr = getLasTlbr(fname);

    //if (fileTlbr(0) > userOuterTlbr(2) or fileTlbr(1) > userOuterTlbr(3) or
        //fileTlbr(2) < userOuterTlbr(0) or fileTlbr(2) < userOuterTlbr(0)) {
    if (false) {
      std::swap(fileNames[ii], fileNames.back());
      fileNames.pop_back();
      std::cout << " - discarding " << fname << "\n";
    } else {

      fileTlbrs.push_back(fileTlbr);

      if (fileTlbr(0) < seenOuterTlbr(0) or seenOuterTlbr(0) == 0) seenOuterTlbr(0) = fileTlbr(0);
      if (fileTlbr(1) < seenOuterTlbr(1) or seenOuterTlbr(1) == 0) seenOuterTlbr(1) = fileTlbr(1);
      if (fileTlbr(2) > seenOuterTlbr(2) or seenOuterTlbr(2) == 0) seenOuterTlbr(2) = fileTlbr(2);
      if (fileTlbr(3) > seenOuterTlbr(2) or seenOuterTlbr(3) == 0) seenOuterTlbr(3) = fileTlbr(3);
    }
  }

  //outerTlbr = Eigen::Vector4f {
    //std::max(seenOuterTlbr(0), userOuterTlbr(0)),
    //std::max(seenOuterTlbr(1), userOuterTlbr(1)),
    //std::min(seenOuterTlbr(2), userOuterTlbr(2)),
    //std::min(seenOuterTlbr(3), userOuterTlbr(3)) };
  outerTlbr = seenOuterTlbr;
  std::cout << " - UTM bounds: " << outerTlbr.transpose() << "\n";
  std::cout << " - Utm wh: " << (outerTlbr.tail<2>()-outerTlbr.head<2>()).transpose() << "\n";

  baseRes = 1. / scaleFactor(baseLvl);
  std::cout << " - Base lvl | res : " << ((int)baseLvl) << " | " << baseRes << "\n";

  lo = utmToTileLo(outerTlbr(0), outerTlbr(1), baseLvl);
  hi = utmToTileHi(outerTlbr(2), outerTlbr(3), baseLvl);


  widthTiles = hi(0) - lo(0);
  heightTiles = hi(1) - lo(1);
  std::cout << " - Grid span: " << lo.transpose() << " -> " << hi.transpose() << " (" << widthTiles << " " << heightTiles << " tiles)\n";
  tiles.resize(heightTiles*widthTiles);

  for (int y=lo(1); y<hi(1); y++)
  for (int x=lo(0); x<hi(0); x++) {
    int idx = (y-lo(1))*widthTiles + (x-lo(0));
    Eigen::Vector4d tlbr = tileTlbr(x,y,baseLvl);
    tiles[idx].tlbr = tlbr;
    tiles[idx].x = x; tiles[idx].y = y; tiles[idx].baseLvl = baseLvl;

    Eigen::AlignedBox<float,2> tileBox(
        tlbr.head<2>().cast<float>().array() - overlap,
        tlbr.tail<2>().cast<float>().array() + overlap);

    for (int ii=0; ii<fileTlbrs.size(); ii++) {
      Eigen::AlignedBox<float,2> fileBox(fileTlbrs[ii].head<2>(), fileTlbrs[ii].tail<2>());
      //std::cout << " - Check " << tileBox.min().transpose() << " " << tileBox.max().transpose()
                               //<< fileBox.min().transpose() << " " << fileBox.max().transpose() << "\n";
      if (tileBox.intersects(fileBox))
        //tiles[idx].srcDsets.push_back(ii);
        tiles[idx].srcDsets.push_back(fileNames[ii]);
    }

    std::cout << " - Tile " << x << " " << y << " has " << tiles[idx].srcDsets.size() << " incident dsets.\n";
  }
}

void LasTile::load(int stride) {
  std::cout << " - Loading tile " << x << " " << y << " (tlbr " << tlbr.transpose() << ")\n";
  for (const std::string& fn : srcDsets) {
    read_las_aoi(fn, tlbr, pts, stride);
  }
  std::cout << " - Loading tile " << x << " " << y << " ... " << pts.size() << " pts.\n";
  for (int i=0; i<pts.size(); i+=pts.size() / 10) {
    //std::cout << " - pt " << pts[i].transpose() << "\n";
    std::cout << " - pt " << pts[i] << "\n";
  }
}
