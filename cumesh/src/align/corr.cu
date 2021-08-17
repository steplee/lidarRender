#include <cufft.h>
#include "helper_cuda.hpp"
#include "cu_image.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

// The images contained are padded.
// fullSquareSize is with overlap, but without padding.
// Actual image sizes are pitch by pitch, which includes half zeros along each axis
struct ImagePatches {
  float *buf;
  int fullSquareSize, overlap, stepSize;
  int n, nw, nh, pitch;

  void release();
};
void ImagePatches::release() { cudaFree(buf); }


static void show_patches(const ImagePatches& ps, const char* name, int wait=0) {
  //int w = ps.nw * ps.fullSquareSize;
  //int h = ps.nh * ps.fullSquareSize;
  //cv::Mat dimg(h,w, CV_32F);
  int s = ps.fullSquareSize;
  int p = ps.pitch;
  for (int yy=0; yy<ps.nh; yy++)
  for (int xx=0; xx<ps.nw; xx++) {
    //cudaMemcpy(dimg(cv::Rect{xx*s,yy*s,s,s}).data, ps.buf+(yy*ps.nw+xx)*s*s, sizeof(float)*s*s, cudaMemcpyDeviceToHost);
    cv::Mat dimg(s,s,CV_32F);
    //cudaMemcpy(dimg.data, ps.buf+(yy*ps.nw+xx)*p*p, sizeof(float)*s*s, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(dimg.data, s*sizeof(float), ps.buf+(yy*ps.nw+xx)*p*p, p*sizeof(float), sizeof(float)*s,s, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    double min, max; cv::minMaxLoc(dimg,&min,&max);
    //std::cout << " - min max " << min << " " << max << "\n";
    cv::normalize(dimg,dimg, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow(name, dimg); cv::waitKey(wait);
  }
  //double min, max; cv::minMaxLoc(dimg,&min,&max);
  //std::cout << " - min max " << min << " " << max << "\n";
  //cv::normalize(dimg,dimg, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  //cv::resize(dimg, dimg, cv::Size{1000, 1000*dimg.rows/dimg.cols});
  //cv::imshow(name, dimg); cv::waitKey(wait);
}

static void show_corr(const ImagePatches& pa, const ImagePatches& pb, const float *corr, const char* name, int wait=0) {
  cudaDeviceSynchronize();
  int s = pa.fullSquareSize;
  int p = pa.pitch;

  cv::Mat dimg(s,3*s,CV_32F);
  cv::Mat dimgs[5];
  dimgs[0] = cv::Mat(p,p,CV_8UC1);
  dimgs[1] = cv::Mat(p,p,CV_8UC1);
  dimgs[2] = cv::Mat(p,p,CV_8UC1);
  dimgs[3] = cv::Mat(p,p,CV_32F);
  dimgs[4] = cv::Mat(p,p,CV_32F);
  for (int yy=0; yy<pa.nh; yy++)
  for (int xx=0; xx<pa.nw; xx++) {
    //cudaMemcpy2D(dimg.data+0*s*sizeof(float), 3*s*sizeof(float), pa.buf+(yy*pa.nw+xx)*p*p, p*sizeof(float), sizeof(float)*s,s, cudaMemcpyDeviceToHost);
    //cudaMemcpy2D(dimg.data+1*s*sizeof(float), 3*s*sizeof(float), pb.buf+(yy*pb.nw+xx)*p*p, p*sizeof(float), sizeof(float)*s,s, cudaMemcpyDeviceToHost);
    //cudaMemcpy2D(dimg.data+2*s*sizeof(float), 3*s*sizeof(float), corr+(yy*pa.nw+xx)*p*p, p*sizeof(float), sizeof(float)*s,s, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(dimgs[3].data, 1*p*sizeof(float), pa.buf+(yy*pa.nw+xx)*p*p, p*sizeof(float), sizeof(float)*p,p, cudaMemcpyDeviceToHost);
    cv::normalize(dimgs[3],dimgs[1], 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cudaMemcpy2D(dimgs[3].data, 1*p*sizeof(float), pb.buf+(yy*pb.nw+xx)*p*p, p*sizeof(float), sizeof(float)*p,p, cudaMemcpyDeviceToHost);
    cv::normalize(dimgs[3],dimgs[2], 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cudaMemcpy2D(dimgs[3].data, 1*p*sizeof(float), corr+(yy*pa.nw+xx)*p*p, p*sizeof(float), sizeof(float)*p,p, cudaMemcpyDeviceToHost);
    // FFT Shift
    //dimgs[3](cv::Rect{0,0,p/2,p/2}).copyTo(dimgs[4](cv::Rect{p/2,p/2,p/2,p/2}));
    //dimgs[3](cv::Rect{p/2,0,p/2,p/2}).copyTo(dimgs[4](cv::Rect{0,p/2,p/2,p/2}));
    //dimgs[3](cv::Rect{0,p/2,p/2,p/2}).copyTo(dimgs[4](cv::Rect{p/2,0,p/2,p/2}));
    //dimgs[3](cv::Rect{p/2,p/2,p/2,p/2}).copyTo(dimgs[4](cv::Rect{0,0,p/2,p/2}));
    dimgs[4] = dimgs[3].clone();


    cv::Point mini,maxi;
    double min, max; cv::minMaxLoc(dimgs[4],&min,&max,&mini,&maxi);
    //std::cout << " - corr min max " << min << " " << max << " | " << maxi << "\n";
    cv::normalize(dimgs[4],dimgs[0], 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::hconcat(dimgs, 3, dimg);

    cv::cvtColor(dimg,dimg, cv::COLOR_GRAY2BGR);
    cv::circle(dimg, maxi, 5, cv::Scalar{0,255,0}, 1);

    cv::imshow(name, dimg); cv::waitKey(wait);
  }
}


__global__ void slice_and_pad_(float* outs, int squaresWide, int squareSize, int overlapSize, int pitch, float* inImg, int w, int h, bool flip) {
  int qy = blockIdx.x;
  int qx = blockIdx.y;

  int fullSquareSize = squareSize + 2 * overlapSize;

  int rowOffset = threadIdx.x + blockIdx.x * (squareSize);
  int colOffset = blockIdx.y * squareSize;

  //int inRowOff = qy * squareSize / (fullSquareSize);
  //int inColOff = qx * squareSize / (fullSquareSize);

  // Apply linear fall-of at edges of big image, and Hamming window on each patch
  float edy = min(1.f, ( min(rowOffset, h - rowOffset)/32.f));
  float ham_y = pow(sin(M_PI*threadIdx.x/fullSquareSize), 2.0);

  for (int c=0; c<fullSquareSize; c++) {
    float v = 0;
    if (colOffset+c < w and rowOffset < h) v = inImg[rowOffset*w+colOffset+c];
    float window_v = 1;

    // Window the original image boundaries!
    float edx = min(1.f, ( min(colOffset+c, w - colOffset-c)/32.f));
    v = v * edx * edy;

    // Window the patch boundaries!
    float ham_x = pow(sin(M_PI*c/fullSquareSize), 2.0);
    v = v * ham_x * ham_y;

    int yy = (pitch-fullSquareSize)/2+threadIdx.x;
    int xx = c + (pitch-fullSquareSize)/2;
    if (flip) {
      yy = pitch - 1 - yy;
      xx = pitch - 1 - xx;
    }
    outs[(qy*squaresWide+qx)*pitch*pitch + (yy)*pitch + xx] = v;
  }
}

ImagePatches slice_and_pad(CuImage<float>& inImg, bool flip) {
  int squareSize = 256, overlapSize = 128/2;
  int fullSquareSize = squareSize + 2 * overlapSize;
  //int inW = 1024, inH = 1024;
  int inW = inImg.w, inH = inImg.h;

  int squaresHigh = (inH + squareSize - 1) / squareSize;
  int squaresWide = (inW + squareSize - 1) / squareSize;

  // Strictly speaking, I should append W zeros along both axes.
  // However, with the hamming window it appears to be good enough
  //int pitch = (fullSquareSize * 1);
  int pitch = (fullSquareSize * 2);

  ImagePatches outs;

  outs.n  = squaresWide * squaresHigh;
  outs.nw = squaresWide;
  outs.nh = squaresHigh;
  outs.fullSquareSize = fullSquareSize;
  outs.overlap = overlapSize;
  outs.pitch = pitch;
  outs.stepSize = squareSize;

  printf(" - Slicing img (%d %d) to get (%d * %d**2) patches.\n", inH,inW, outs.n, outs.fullSquareSize);

  cudaMalloc(&outs.buf, pitch*pitch*squaresWide*squaresHigh*sizeof(float));
  cudaMemset(outs.buf, 0,pitch*pitch*squaresWide*squaresHigh*sizeof(float));

  cudaDeviceSynchronize();

  dim3 blk(squaresHigh, squaresWide);
  dim3 thr(fullSquareSize);
  getLastCudaError("pre");
  slice_and_pad_<<<blk,thr>>>(outs.buf, squaresWide, squareSize, overlapSize, pitch, inImg.buf,inImg.w,inImg.h,flip);
  getLastCudaError("post slice");

  return outs;
}

__device__ float2 cmplx_mult(const float2& u, const float2& v, const float& scale) {
  return make_float2(
      scale * (u.x*v.x - u.y*v.y),
      scale * (u.x*v.y + u.y*v.x)
  );
}

#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>
void multiply_spectra(float2* out, const float2* a, const float2* b, int l, int n) {
  float scale = 1.f / n;
  auto it = thrust::make_zip_iterator(thrust::make_tuple(a,b));
  thrust::transform(thrust::device, it, it+l*n, out,
      [=]__device__(const auto& a) {
        return cmplx_mult(thrust::get<0>(a), thrust::get<1>(a), scale);
      });
}

void fft_shift(float* out, float* corrs, int n, int w, int h) {
  auto it = thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
      [=]__device__(const int& i) {
        int b = i / (w*h);
        int y = (i % (w*h)) / w;
        int x = (i % (w*h)) % w;
        y = (y - h / 2);
        x = (x - w / 2);
        if (y < 0) y += h;
        if (x < 0) x += w;
        return b*w*h + y*w + x;
      });
  thrust::gather(thrust::device, it, it+n*w*h, corrs, out);
}

void fft_get_max_location(int2* outs, float* corrs, int n, int w, int h) {
  thrust::tuple<int,float>* tmp;
  cudaMallocManaged(&tmp, sizeof(thrust::tuple<int,float>)*n);

  auto it = thrust::make_zip_iterator(thrust::make_tuple(
      thrust::counting_iterator<int>(0),
      corrs));
  auto it2 = thrust::reduce_by_key(thrust::device, it, it+n*w*h,
      it, thrust::make_discard_iterator(), tmp,
      [=]__device__(const auto& a, const auto& b) {
        return thrust::get<0>(a) / (w*h) == thrust::get<0>(b) / (w*h);
      },
      [=]__device__(const auto& a, const auto& b) {
        //printf(" - comparing %f %f\n", thrust::get<1>(a) , thrust::get<1>(b));
        return thrust::get<1>(a) > thrust::get<1>(b) ? a : b;
      });
  //std::cout << " - Had " << it2.second - tmp << " keys.\n";

  cudaDeviceSynchronize();
  for (int i=0; i<n; i++) {
    //std::cout << " - ind " << thrust::get<0>(tmp[i])
      //<< " i " << thrust::get<0>(tmp[i]) % (w*h)
      //<< " x " << (thrust::get<0>(tmp[i]) % (w*h)) % w
      //<< " y " << (thrust::get<0>(tmp[i]) % (w*h)) / w
      //<< " wh " << w << " " << h
      //<< "\n";
    int xx = (thrust::get<0>(tmp[i]) % (w*h)) % w - w/2;
    int yy = (thrust::get<0>(tmp[i]) % (w*h)) / w - h/2;
    outs[i] = make_int2(xx,yy);
  }
  cudaFree(tmp);
}

#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
                fprintf(stderr, "CUFFT error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, \
                           _cudaGetErrorEnum(err)); \
             cudaDeviceReset(); assert(0); \
    }
}

void do_corr(CuImage<float>& imga, CuImage<float>& imgb) {

  ImagePatches patchesa = slice_and_pad(imga, false);
  ImagePatches patchesb = slice_and_pad(imgb,  true);

  cudaDeviceSynchronize();
  show_patches(patchesa, "Patches", 1);
  show_patches(patchesb, "Patches", 1);
  std::cout << " - Showed Patches." << std::endl;

  //int size = patchesa.fullSquareSize;
  int size = patchesa.pitch;

  cufftHandle planFwd, planInv;
  //checkCudaErrors(cufftPlan2d(&fftPlanFwd, size, size, CUFFT_R2C));
  //checkCudaErrors(cufftPlan2d(&fftPlanInv, size, size, CUFFT_C2R));
  int sizes[2] = { size, size };
  size_t workSize1, workSize2;
  size_t batch = patchesa.n;
  int inembed[2] = {size, size};
  int onembed[2] = {size, size};
  int istride=1, ostride=1, idist=size*size, odist=size*size;
  std::cout << " - Making plans." << std::endl;
  //cufftSafeCall(cufftMakePlanMany(planFwd, 2, sizes, inembed,istride,idist, onembed,ostride,odist, CUFFT_R2C, batch, &workSize1));
  //cufftSafeCall(cufftMakePlanMany(planInv, 2, sizes, inembed,istride,idist, onembed,ostride,odist, CUFFT_R2C, batch, &workSize2));
  cufftSafeCall(cufftPlanMany(&planFwd, 2, sizes, inembed,istride,idist, onembed,ostride,odist, CUFFT_R2C, batch));
  cufftSafeCall(cufftPlanMany(&planInv, 2, sizes, inembed,istride,idist, onembed,ostride,odist, CUFFT_C2R, batch));
  std::cout << " - Made plans." << std::endl;

  float2 *fa, *fb, *fab;
  int n = patchesa.n;
  int l = size*size;
  checkCudaErrors(cudaMalloc(&fa, sizeof(float2)*l*n));
  checkCudaErrors(cudaMalloc(&fb, sizeof(float2)*l*n));
  checkCudaErrors(cudaMalloc(&fab, sizeof(float2)*l*n));
  checkCudaErrors(cudaMemset(fab, 0, sizeof(float2)*l*n));
  std::cout << " - Allocated buffers." << std::endl;


  cufftSafeCall(cufftExecR2C(planFwd, (cufftReal*)patchesa.buf, (cufftComplex*)fa));
  cufftSafeCall(cufftExecR2C(planFwd, (cufftReal*)patchesb.buf, (cufftComplex*)fb));
  std::cout << " - Exec'ed ffts." << std::endl;
  multiply_spectra(fab, fa, fb, l, n);
  std::cout << " - Multiplied spectra." << std::endl;
  //cudaMemset(fa, 0,patchesa.pitch*patchesa.pitch*patchesa.nw*patchesa.nh*sizeof(float));
  cufftSafeCall(cufftExecC2R(planInv, (cufftComplex*)fab, (cufftReal*)fab));
  std::cout << " - Exec'ed ifft." << std::endl;

  float* ifab = (float*)fa;
  fft_shift(ifab, (float*)fab, n, size, size);

  //show_corr(patchesa, patchesb, (float*)ifab, "corr", 0);
  int2 maxLocations[n];
  fft_get_max_location(maxLocations, (float*)ifab, n, size, size);
  //for (int i=0; i<n; i++) std::cout << " - max at " << maxLocations[i].x << " " << maxLocations[i].y << "\n";

  // Show offsets
  cv::Mat dimg0(imga.h,imga.w,CV_32F), dimga, dimgb;
  cudaMemcpy(dimg0.data, imga.buf, sizeof(float)*imga.w*imga.h, cudaMemcpyDeviceToHost);
  cv::normalize(dimg0,dimga, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cudaMemcpy(dimg0.data, imgb.buf, sizeof(float)*imgb.w*imgb.h, cudaMemcpyDeviceToHost);
  cv::normalize(dimg0,dimgb, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::Mat dimg(imga.h, imga.w, CV_8UC3);
  for (int y=0; y<imga.h; y++)
  for (int x=0; x<imga.w; x++) {
    uint8_t b = dimga.at<uint8_t>(y,x);
    uint8_t g = dimgb.at<uint8_t>(y,x);
    dimg.at<cv::Vec3b>(y,x) = cv::Vec3b{b,g,0};
  }
  for (int yy=0; yy<patchesa.nh; yy++)
  for (int xx=0; xx<patchesa.nw; xx++) {
    int i = yy*patchesa.nw + xx;
    int y0 = yy * patchesa.stepSize + patchesa.stepSize / 2 + maxLocations[i].x;
    int x0 = xx * patchesa.stepSize + patchesa.stepSize / 2 + maxLocations[i].y;
    cv::Point pt0 { y0 , x0 };

    for (int dy=-1; dy<1; dy++)
    for (int dx=-1; dx<1; dx++) {
      if (dy == 0 and dx == 0) continue;
      if (dy == dx) continue;
      if (x0+dx > 0 and y0+dy > 0) {
        int j = (yy+dy)*patchesa.nw + (dx+xx);
        int y1 = (yy + dy) * patchesa.stepSize + patchesa.stepSize / 2 + maxLocations[j].x;
        int x1 = (xx + dx) * patchesa.stepSize + patchesa.stepSize / 2 + maxLocations[j].y;
        cv::Point pt1 { y1 , x1 };
        std::cout << " - line " << y0 << " " << x0 << " " << pt0 << " " << pt1 << " " << patchesa.stepSize << "\n";
        cv::line(dimg, pt0, pt1, cv::Scalar{0,255,0}, 1);
      }
    }
  }
  //cv::resize(dimg,dimg, cv::Size{1024, 1024*dimg.rows/dimg.cols});
  cv::pyrDown(dimg,dimg);
  cv::imshow("Grid", dimg); cv::waitKey(0);

  cudaFree(fab);
  cudaFree(fb);
  cudaFree(fa);
  patchesa.release();
  patchesb.release();

}
