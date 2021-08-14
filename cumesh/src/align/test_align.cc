

#include "align.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

static void show_img(const CuImage<float>& img, const char* name, int wait=0, int fixW=-1,int fixH=-1) {
  int w = fixW == -1 ? img.w : fixW;
  int h = fixH == -1 ? img.h : fixH;
  cv::Mat dimg(h,w, CV_32F);
  cudaMemcpy(dimg.data, img.buf, sizeof(float)*w*h, cudaMemcpyDeviceToHost);
  double min, max; cv::minMaxLoc(dimg,&min,&max);
  cv::normalize(dimg,dimg, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::imshow(name, dimg); cv::waitKey(wait);
}

int main() {

  //cv::Mat src(1024,1024, CV_8UC1);
  //src = cv::Scalar{0};
  //src(cv::Rect(200,200,200,200)) = cv::Scalar{100};
  //cv::Mat src = cv::imread("/home/slee/Downloads/IMG_20200525_212447.jpg", 0);
  cv::Mat src = cv::imread("/home/slee/Downloads/IMG_20190824_101637.jpg", 0);
  int h = src.rows, w = src.cols;
  cv::resize(src,src, cv::Size{w,h});
  cv::GaussianBlur(src,src, cv::Size{7,7}, 1);
  cv::GaussianBlur(src,src, cv::Size{7,7}, 1);
  cv::GaussianBlur(src,src, cv::Size{7,7}, 1);

  //cv::Mat elevMat(1024,1024, CV_32FC1);
  //elevMat = cv::Scalar{0};
  //elevMat(cv::Rect(200,200,200,200)) = cv::Scalar{100};
  cv::Mat elevMat;
  float HH[9] = {
    1.001,0,20,
    0,1,0,
    //.99,.1,-5,
    //-.1,.99,-20,
    0,0,1
  };
  cv::Mat H(3,3,CV_32F,HH);
  cv::warpPerspective(src,elevMat, H, cv::Size{elevMat.cols,elevMat.rows});
  elevMat.convertTo(elevMat, CV_32FC1, .5 / 255.);

  CuImage<float> elev;
  elev.allocate(w,h,1);
  cudaMemcpy(elev.buf, elevMat.data, sizeof(float)*w*h, cudaMemcpyHostToDevice);

  //show_img(elev, "elevInput", 0);
  align_tiff(elev, src);

  return 0;
}
