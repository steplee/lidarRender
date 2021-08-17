#include "cu_image.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

#include "align.h"

static void show_img(const CuImage<float>& img, const char* name, int wait=0, int fixW=-1,int fixH=-1) {
  int w = fixW == -1 ? img.w : fixW;
  int h = fixH == -1 ? img.h : fixH;
  cv::Mat dimg(h,w, CV_32F);
  cudaMemcpy(dimg.data, img.buf, sizeof(float)*w*h, cudaMemcpyDeviceToHost);
  double min, max; cv::minMaxLoc(dimg,&min,&max);
  std::cout << " - min max " << min << " " << max << "\n";
  cv::normalize(dimg,dimg, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::imshow(name, dimg); cv::waitKey(wait);
}

//constexpr int k = 3;
constexpr int blkSize = 16;
__global__ void compute_grad_norm_(
    float* grad, int ow, int oh,
    float* elev, int iw, int ih) {
  //int k2 = k/2;
  //int ty = threadIdx.x-k2 + (blockIdx.x-k2) * blkSize;
  //int tx = threadIdx.y-k2 + (blockIdx.y-k2) * blkSize;
  int ty = threadIdx.x + (blockIdx.x) * (blkSize);
  int tx = threadIdx.y + (blockIdx.y) * (blkSize);
  int ly = threadIdx.x;
  int lx = threadIdx.y;

  constexpr int N = blkSize + 2;
  __shared__ float fetched[N*N];

  //for (int dy=-k2; dy<=k2; dy++)
  //for (int dx=-k2; dx<=k2; dx++) {
    //int y = ty + dy, x = tx + dx;
    //int lly = ly + dy, llx = lx + dx;
  {
    int y = ty, x = tx;
    if (y>=0 and y<ih and x>=0 and x<iw) {
      fetched[ly*blockDim.y+lx] = elev[ty*iw+tx];
    } else
      fetched[ly*blockDim.y+lx] = 0;
  }

  __syncthreads();

  if (ly>=1 and ly<blkSize+1 and lx>=1 and lx<blkSize+1 and
      ty>=0  and tx>=0         and ty<oh  and tx<ow) {
    float dx, dy;
    dx = fetched[(ly-1)*blockDim.y+(lx-1)]     +
         fetched[(ly  )*blockDim.y+(lx-1)] * 2 +
         fetched[(ly+1)*blockDim.y+(lx-1)]     + -(
         fetched[(ly-1)*blockDim.y+(lx+1)]     +
         fetched[(ly  )*blockDim.y+(lx+1)] * 2 +
         fetched[(ly+1)*blockDim.y+(lx+1)]);
    dy = fetched[(ly-1)*blockDim.y+(lx-1)]     +
         fetched[(ly-1)*blockDim.y+(lx  )] * 2 +
         fetched[(ly-1)*blockDim.y+(lx+1)]     + -(
         fetched[(ly+1)*blockDim.y+(lx-1)]     +
         fetched[(ly+1)*blockDim.y+(lx  )] * 2 +
         fetched[(ly+1)*blockDim.y+(lx+1)]);
    dx /= 4;
    dy /= 4;
    float mag = sqrtf(dx * dx + dy * dy);
    grad[ty*ow+tx] = mag;
    //printf(" - mag %f\n", mag);
    //grad[ty*ow+tx] = fetched[ly*blockDim.y+lx];
  }

}

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

void compute_grad_norm(CuImage<float>& grad, CuImage<float>& elev) {
  int h = elev.h, w = elev.w;
  grad.allocate(w,h,1);
  cudaMemset(grad.buf, 0, sizeof(float)*w*h);

  dim3 blk((h+blkSize-1)/blkSize, (w+blkSize-1)/blkSize);
  dim3 thr(blkSize+2,blkSize+2);
  compute_grad_norm_<<<blk,thr>>>(grad.buf, w,h, elev.buf,w,h);
  cudaDeviceSynchronize();

  /*
  float mean = thrust::reduce(thrust::device, grad.buf, grad.buf+w*h, 0.f,
      []__device__(const float& a, const float& b) { return (a+b)/2.f; });
  thrust::transform(thrust::device, grad.buf, grad.buf+w*h, grad.buf,
      [=]__device__(const float& a) { return a-mean; });
      */

  show_img(grad, "GradNorm", 0);
}

void align_it() {
}

void align_tiff(
    CuImage<float>& elev,
    cv::Mat img_) {

  CuImage<float> elevGrad;
  compute_grad_norm(elevGrad, elev);

  CuImage<float> img;
  CuImage<float> rgbGrad;
  cv::Mat img__;
  img_.convertTo(img__, CV_32FC1, 1./255.);
  int h = img_.rows, w = img_.cols;
  img.allocate(w,h,1);
  cudaMemcpy(img.buf, img__.data, sizeof(float)*h*w, cudaMemcpyHostToDevice);
  compute_grad_norm(rgbGrad, img);

  do_corr(rgbGrad, elevGrad);
}
