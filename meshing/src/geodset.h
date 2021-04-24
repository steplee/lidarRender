#pragma once
#include <Eigen/StdVector>
#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <gdal_priv.h>

class GDALDataset;
class OGRCoordinateTransformation;
class GDALRasterBand;

using RowMatrix53 = Eigen::Matrix<double,5,3,Eigen::RowMajor>;
using RowMatrixX3 = Eigen::Matrix<double,-1,3,Eigen::RowMajor>;
using Vector4d = Eigen::Vector4d;
using Vector3d = Eigen::Vector3d;
using Vector2d = Eigen::Vector2d;


// Thin wrapper around GDALDataset
class SimpleGeoDataset {
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SimpleGeoDataset(const std::string& name);
    ~SimpleGeoDataset();

  bool bboxNative(Vector4d& bboxNative, int outw, int outh, cv::Mat& out);
  bool tlbrNative(Vector4d& tlbr, int outw, int outh, cv::Mat& out);
  bool bboxPix(Vector4d& bboxPix, int outw, int outh, cv::Mat& out);

  //protected:

  Eigen::Matrix<double,2,3,Eigen::RowMajor> pix2native;
  Eigen::Matrix<double,2,3,Eigen::RowMajor> native2pix;
  Vector2d pix2gps(double x, double y);

  //inline RowMatrixXd ...

  GDALDataset* dset = nullptr;
  OGRCoordinateTransformation* wm2native = nullptr;
  OGRCoordinateTransformation* native2wm = nullptr;
  OGRCoordinateTransformation* native2gps = nullptr;
  OGRCoordinateTransformation* gps2native = nullptr;
  int nbands; // Either 3 or 1
  int cv_type; // Either CV_8U or CV_8UC3
  size_t eleSize = 1;
  int fullWidth=0, fullHeight=0;

  GDALDataType gdalType;
  bool myRasterIO(int xoff, int yoff, int xsize, int ysize, void* buf, int outw, int outh);

  char* myBuf = 0;
  int myBufSize = 0;
  Vector4d tlbr_native;

  // bboxNative ...

  int numBands=0;
  GDALRasterBand* bands[4];
  bool block(cv::Mat& dst, int y, int x, int lvl);
  int        getNumBlocksX(int lvl) { return numBlocksPerLvl[lvl](1); }
  int        getNumBlocksY(int lvl) { return numBlocksPerLvl[lvl](0); }
  int        getBlockSizeX() { return blockSizeX; }
  int        getBlockSizeY() { return blockSizeY; }
  inline int getNumLevels() { return numLevels; }
  int                          numLevels, blockSizeX, blockSizeY;
  std::vector<Eigen::Vector2i> numBlocksPerLvl;
  Eigen::Vector4d blockPix(int y, int x, int lvl);

  std::string fname;
  std::string getDatasetName();

  SimpleGeoDataset* clone();
};

