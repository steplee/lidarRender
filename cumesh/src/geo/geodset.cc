#include "geodset.h"

#include <iostream>

#include <gdal_priv.h>
#include <ogr_core.h>
#include <ogr_spatialref.h>
#include <opencv2/imgproc.hpp>
#include <Eigen/LU>

using Vector2i = Eigen::Matrix<int,2,1>;

using RowMatrix3d = Eigen::Matrix<double,3,3,Eigen::RowMajor>;

//__attribute__((force_align_arg_pointer))
SimpleGeoDataset::SimpleGeoDataset(const std::string& name) {

  GDALAllRegister();

  fname = name;
  dset = (GDALDataset*) GDALOpen(name.c_str(), GA_ReadOnly);
  double g[6];
  dset->GetGeoTransform(g);
  RowMatrix3d pix2native_; pix2native_ << g[1], g[2], g[0], g[4], g[5], g[3], 0, 0, 1;
  RowMatrix3d native2pix_ = pix2native_.inverse();
  pix2native = pix2native_.topRows<2>();
  native2pix = native2pix_.topRows<2>();
  fullWidth  = dset->GetRasterXSize();
  fullHeight = dset->GetRasterYSize();

  //numBands = dset->GetRasterCount();
  nbands = dset->GetRasterCount() >= 3 ? 3 : 1;
  auto band = dset->GetRasterBand(0 + 1);
  for (int i=0; i<nbands; i++) bands[i] = dset->GetRasterBand(1+i);

  gdalType = dset->GetRasterBand(1)->GetRasterDataType();
  if (not (gdalType == GDT_Byte or gdalType == GDT_Int16 or gdalType == GDT_Float32)) {
    std::cerr << " == ONLY uint8_t/int16_t/float32 dsets supported right now." << std::endl;
    exit(1);
  }
  if (nbands == 3 and gdalType == GDT_Byte) cv_type = CV_8UC3, eleSize = 1;
  if (nbands == 1 and gdalType == GDT_Byte) cv_type = CV_8UC1, eleSize = 1;
  if (nbands == 1 and gdalType == GDT_Int16) cv_type = CV_16SC1, eleSize = 2;
  if (nbands == 1 and gdalType == GDT_Float32) cv_type = CV_32FC1, eleSize = 4;

  OGRSpatialReference sr_native, sr_3857, sr_4326;
  char* pp = const_cast<char*>(dset->GetProjectionRef());
  sr_native.importFromWkt(&pp);
  sr_3857.importFromEPSG(3857);
  sr_4326.importFromEPSG(4326);
  wm2native = OGRCreateCoordinateTransformation(&sr_3857, &sr_native);
  native2wm = OGRCreateCoordinateTransformation(&sr_native, &sr_3857);
  gps2native = OGRCreateCoordinateTransformation(&sr_4326, &sr_native);
  native2gps = OGRCreateCoordinateTransformation(&sr_native, &sr_4326);

  // Set tlbr_wm.
  tlbr_native.head<2>() = pix2native * Vector3d{ 0 , 0 , 1 };
  tlbr_native.tail<2>() = pix2native * Vector3d{ (double)fullWidth , (double)fullHeight , 1 };

  numLevels = band->GetOverviewCount() + 1;
  band->GetBlockSize(&blockSizeX, &blockSizeY);
  for (int lvl = 0; lvl < numLevels; lvl++)
      if (lvl > 0)
          numBlocksPerLvl.push_back(Vector2i{
              (dset->GetRasterBand(1)->GetOverview(lvl - 1)->GetYSize() + blockSizeY - 1) / blockSizeY,
              (dset->GetRasterBand(1)->GetOverview(lvl - 1)->GetXSize() + blockSizeX - 1) / blockSizeX });
      else
          numBlocksPerLvl.push_back(Vector2i{
              (dset->GetRasterBand(1)->GetYSize() + blockSizeY - 1) / blockSizeY,
              (dset->GetRasterBand(1)->GetXSize() + blockSizeX - 1) / blockSizeX });
}
SimpleGeoDataset::~SimpleGeoDataset() {
  if (wm2native) OCTDestroyCoordinateTransformation(wm2native);
  if (native2wm) OCTDestroyCoordinateTransformation(native2wm);
  if (native2gps) OCTDestroyCoordinateTransformation(native2gps);
  if (gps2native) OCTDestroyCoordinateTransformation(gps2native);
  if (dset) GDALClose(dset);
  if (myBuf) {
    myBufSize = 0;
    CPLFree(myBuf);
    myBuf = 0;
  }
}

bool SimpleGeoDataset::block(cv::Mat& dst, int y, int x, int lvl) {
  dst.create(blockSizeY, blockSizeX, cv_type);

  GDALRasterBand *bands_[4]; // Max 4 bands
  if (lvl > 0) for (int b = 0; b < nbands; b++) bands_[b] = bands[b]->GetOverview(lvl - 1);
  else for (int b = 0; b < nbands; b++) bands_[b] = bands[b];

  int sy = blockSizeY, sx = blockSizeX;
  //assert(sy == 256);
  if (myBufSize < sy*sx*eleSize*nbands) {
    if (myBuf) CPLFree(myBuf);
    myBufSize = sy*sx*eleSize*nbands;
    myBuf = (char*) CPLMalloc(myBufSize);
    std::cout << " - ALLOCATED " << myBufSize << " at " << myBuf << std::endl;
    assert(myBuf != nullptr);
  }
  //char myBuf[256*256*4];

  for (int b = 0; b < nbands; b++) {
    auto stat = bands_[b]->ReadBlock(x, y, myBuf + eleSize * sy * sx * b);
      if (stat != CE_None) {
        // If we fail to read a block, make it black.
        memset(myBuf+eleSize*sy*sx*b, 0, eleSize*sy*sx);
      }
  }

    const auto g_off = sx*sy;
    const auto b_off = sx*sy*2;
    const auto w = dst.cols;
    if (cv_type == CV_8UC3) {
        uint8_t *buf_ = (uint8_t *)(myBuf);
        // uint8_t* ptr_ = dst.ptr();
        for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++) {
                dst.data[(y)*w*3 + (x)*3 + 0] = buf_[y * sx + x];
                dst.data[(y)*w*3 + (x)*3 + 1] = buf_[y * sx + x + g_off];
                dst.data[(y)*w*3 + (x)*3 + 2] = buf_[y * sx + x + b_off];
            }
    } else if (cv_type == CV_8UC1 or cv_type == CV_16SC1) {
        uint8_t *buf_ = (uint8_t *)(myBuf);
        memcpy(dst.data, buf_, eleSize*dst.cols*dst.rows);
    } else if (cv_type == CV_32FC1) {
        uint8_t *buf_ = (uint8_t *)(myBuf);
        memcpy(dst.data, buf_, eleSize*dst.cols*dst.rows);
    }

    return true;

}

Eigen::Vector4d SimpleGeoDataset::blockPix(int y, int x, int lvl) {
    Eigen::Vector4d out;
    int             scale = 1 << lvl;
    int             w = blockSizeX * scale, h = blockSizeY * scale;
    out(0) = x * w;
    out(1) = y * h;
    out(2) = w;
    out(3) = h;
    return out;
}
Vector2d SimpleGeoDataset::pix2gps(double x, double y) {
    Vector2d xy = pix2native * Vector3d{x, y, 1};
    native2gps->Transform(1, &(xy(0)), &(xy(1)));
    return xy;
}

std::string SimpleGeoDataset::getDatasetName() {
  return fname;
}

bool SimpleGeoDataset::myRasterIO(int xoff, int yoff, int xsize, int ysize, void* buf, int outw, int outh) {
  return dset->RasterIO(GF_Read, xoff,yoff,xsize,ysize, buf, outw,outh,
      (GDALDataType)gdalType, nbands,nullptr,
      nbands,eleSize*nbands*outw,1, nullptr);
}

bool SimpleGeoDataset::tlbrNative(Vector4d& tlbr, int outw, int outh, cv::Mat& out) {
  Vector4d bbox { tlbr(0), tlbr(1), tlbr(2)-tlbr(0), tlbr(3)-tlbr(1) };
  return bboxNative(bbox, outw,outh,out);
}
bool SimpleGeoDataset::bboxNative(Vector4d& bboxNative, int outw, int outh, cv::Mat& out) {
  out.create(outh, outw, cv_type);

  Vector2d tl = (native2pix * Vector3d{bboxNative(0),bboxNative(1),1.});
  Vector2d br = (native2pix * Vector3d{bboxNative(2)+bboxNative(0),bboxNative(3)+bboxNative(1),1.});
  if (tl(0) > br(0)) std::swap(tl(0),br(0));
  if (tl(1) > br(1)) std::swap(tl(1),br(1));
  int xoff = tl(0);
  int yoff = tl(1);
  int xsize = (int) (br(0) - tl(0));
  int ysize = (int) (br(1) - tl(1));

  if (xoff > 0 and xoff+xsize < fullWidth and yoff > 0 and yoff+ysize < fullHeight) {
    // normal case: query box strictly lies inside
    //auto good = myRasterIO(xoff,yoff,xsize,ysize, out.data, outw,outh);
    //return good == true;

    GDALRasterIOExtraArg arg;
    arg.nVersion = RASTERIO_EXTRA_ARG_CURRENT_VERSION;
    arg.eResampleAlg = GRIORA_Bilinear;
    arg.pfnProgress = 0;
    arg.pProgressData = 0;
    arg.bFloatingPointWindowValidity = 0;
    auto err = dset->RasterIO(GF_Read, xoff, yoff, xsize, ysize, out.data, outw,outh,
        gdalType, nbands,nullptr,
        eleSize*nbands,eleSize*nbands*outw,eleSize, &arg);
    return err == CE_None;
  } else if (xoff+xsize >= 0 and xoff <= fullWidth and yoff+ysize >= 0 and yoff <= fullHeight) {
    // case where there is partial overlap

    // NOTE: TODO I haven't really verified this is correct!
    // TODO: Haven't tasted non-unit aspect ratios

    Eigen::Vector4i inner { std::max(0,xoff) , std::max(0,yoff), std::min(fullWidth-1,xoff+xsize), std::min(fullHeight-1,yoff+ysize) };
    float sx = ((float)outw) / xsize;
    float sy = ((float)outh) / ysize;
    int inner_w = inner(2) - inner(0), inner_h = inner(3) - inner(1);
    int read_w = (inner(2)-inner(0)) * sx, read_h = (inner(3)-inner(1)) * sy;
    cv::Mat buf(read_h, read_w, cv_type);
    auto err = dset->RasterIO(GF_Read, inner(0), inner(1), inner_w, inner_h, buf.data, read_w,read_h,
        gdalType, nbands,nullptr,
        eleSize*nbands,eleSize*nbands*read_w,eleSize*1, nullptr);
    if (err != CE_None) {
      return false;
    }

    float in_pts[6]  = { 0,0, sx*inner_w, 0, 0, sy*inner_h };
    float out_pts[6] = {
      sx*(inner(0)-xoff), sy*(inner(1)-yoff),
      sx*(inner(2)-xoff), sy*(inner(1)-yoff),
      sx*(inner(0)-xoff), sy*(inner(3)-yoff)
    };
    /*std::cout << " - s: " << sx << " " << sy << "   " << read_w << " " << read_h << "\n";
    std::cout << " - inPts: "
      << in_pts[0] << " " << in_pts[1] << " " << in_pts[2] << " "
      << in_pts[3] << " " << in_pts[4] << " " << in_pts[5] << "\n";
    std::cout << " - outPts: "
      << out_pts[0] << " " << out_pts[1] << " " << out_pts[2] << " "
      << out_pts[3] << " " << out_pts[4] << " " << out_pts[5] << "\n";*/

    cv::Mat in_pts_(3,2,CV_32F, in_pts);
    cv::Mat out_pts_(3,2,CV_32F, out_pts);
    cv::Mat A = cv::getAffineTransform(in_pts_, out_pts_);
    cv::warpAffine(buf, out, A, cv::Size{outw,outh});
    return true;
  } else {
    out = cv::Scalar{0};
    return false;
  }
}
bool SimpleGeoDataset::bboxPix(Vector4d& bboxPix, int outw, int outh, cv::Mat& out) {
  out.create(outh, outw, cv_type);
  /*int xoff = bboxPix(0);
  int yoff = bboxPix(1);
  int xsize = (int) bboxPix(2);
  int ysize = (int) bboxPix(3);*/
  int xoff = std::min(bboxPix(0), bboxPix(2));
  int yoff = std::min(bboxPix(1), bboxPix(3));
  int xsize = std::max(bboxPix(0), bboxPix(2)) - xoff;
  int ysize = std::max(bboxPix(1), bboxPix(3)) - yoff;
  auto err = dset->RasterIO(GF_Read, xoff, yoff, xsize, ysize, out.data, outw,outh,
      gdalType, nbands,nullptr,
      eleSize*nbands,nbands*outw*eleSize,eleSize, nullptr);
  return err == CE_None;
}



SimpleGeoDataset* SimpleGeoDataset::clone() {
  return new SimpleGeoDataset(getDatasetName());
}
