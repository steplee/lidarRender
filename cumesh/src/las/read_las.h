#pragma once

#include <Eigen/StdVector>
#include <Eigen/Core>

#include "types.h"
//struct LasPoint { float x,y,z; };
//using LasPoint = Eigen::Vector3f;
//struct alignas(128) LasPoint { float x,y,z; };

// 'cnc' = ChartNormalizedCoordinates
// Conversion is actually really simple, thanks to UTM
// having the origin at the bottom left corner for both North and South zones.
// https://www.e-education.psu.edu/natureofgeoinfo/book/export/html/1696
constexpr double UtmSliceWidthMeters  = 666'000;
constexpr double UtmSliceHeightMeters = 10'000'000;
inline Eigen::Vector2d utm_to_cnc(const Eigen::Vector2d& utm) {
  return utm / UtmSliceHeightMeters;
}
inline Eigen::Vector2d cnc_to_utm(const Eigen::Vector2d& cnc) {
  return cnc * UtmSliceHeightMeters;
}
//inline double scaleFactor(const int& lvl) { return 1. / (1<<lvl); }
inline double scaleFactor(const int& lvl) { return (1<<lvl) / UtmSliceHeightMeters; }
inline double scaleFactor1(const int& lvl) { return (1<<lvl); }
inline double invScaleFactor(const int& lvl) { return (1<<lvl); }
inline Eigen::Vector4d tileTlbr(int x, int y, int lvl) {
  return Eigen::Vector4d {
      x / scaleFactor(lvl), y / scaleFactor(lvl),
      (x+1) / scaleFactor(lvl), (y+1) / scaleFactor(lvl) };
}
inline Eigen::Matrix<int,2,1> utmToTileLo(double x, double y, int lvl) {
  return Eigen::Matrix<int,2,1> {
    std::floor(x * scaleFactor(lvl)),
    std::floor(y * scaleFactor(lvl)) };
}
inline Eigen::Matrix<int,2,1> utmToTileHi(double x, double y, int lvl) {
  return Eigen::Matrix<int,2,1> {
    std::ceil(x * scaleFactor(lvl)),
    std::ceil(y * scaleFactor(lvl)) };
}

Eigen::Vector4f getLasTlbr(const std::string &fname);

//std::vector<LasPoint> read_las(const std::string& fname);

// Note: This function returns floating point numbers with origin as TOP-LEFT.
// Single precision floats ARE NOT SUFFICIENT for UTM values. The values are too large.
void read_las_aoi(const std::string& fname, const Eigen::Vector4d& aoiTlbr, std::vector<LasPoint>& out, int stride=1);
std::vector<std::string> find_las_in_box(const Eigen::Vector4f& webMercatorTlbr);

struct LasTile {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::vector<std::string> srcDsets;
  Eigen::Vector4d tlbr;

  int32_t zone;
  //int64_t xyz;
  int32_t x;
  int32_t y;
  int32_t baseLvl;

  std::vector<LasPoint> pts;
  void load(int stride=1);
};


struct MultiLasDataset {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MultiLasDataset(const std::vector<std::string>& dirs);

  int8_t baseLvl = 13;
  double baseRes = -1;


  std::vector<std::string> dirs;
  //Eigen::Vector4f webMercatorTlbr;

  float overlap = 0;

  // Read headers, determine which tiles read which datasets
  void split();

  inline int widthInTiles() { return widthTiles; }
  inline int heightInTiles() { return heightTiles; }

  // Return thin data structure. User should then call tile.load() to load the data.
  LasTile getTile(int x, int y) { return tiles[(y-lo(1))*widthTiles + x-lo(0)]; }
  LasTile getRelativeTile(int x, int y) { return tiles[(y)*widthTiles + x]; }

  private:
    int widthTiles;
    int heightTiles;
    Eigen::Matrix<int,2,1> lo, hi;

    Eigen::Vector4f outerTlbr;
    Eigen::Vector4f seenOuterTlbr;

    // Aligned as structure-of-arrays.
    std::vector<std::string> fileNames;
    std::vector<Eigen::Vector4f> fileTlbrs;

    std::vector<LasTile> tiles;

};

