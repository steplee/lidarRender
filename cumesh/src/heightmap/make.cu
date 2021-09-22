
#include <vector>
#include "binSearch4.cuh"
#include "helper_cuda.hpp"
#include "cu_image.hpp"
#include "make.h"

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
template <class T>
using DevVector = thrust::device_vector<T>;

#include <opencv2/highgui.hpp>

#include "octree.h"
#include "make_geo.h"


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
static void show_img(const CuImage<float2>& img, const char* name, int wait=0, int fixW=-1,int fixH=-1) {
  int w = fixW == -1 ? img.w : fixW;
  int h = fixH == -1 ? img.h : fixH;
  cv::Mat dimg(h,w, CV_32F);
  float* img2; cudaMalloc(&img2, w*h*sizeof(float));
  //thrust::transform(thrust::device, img.buf, img.buf+w*h, img2, []__device__(const float2& f) { return f.y; });
  thrust::transform(thrust::device, img.buf, img.buf+w*h, img2, []__device__(const float2& f) { return f.x; });
  cudaMemcpy(dimg.data, img2, sizeof(float)*w*h, cudaMemcpyDeviceToHost);
  double min, max; cv::minMaxLoc(dimg,&min,&max);
  std::cout << " - min max " << min << " " << max << "\n";
  cv::normalize(dimg,dimg, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::imshow(name, dimg); cv::waitKey(wait);
  cudaFree(img2);
}

// Drop any points below zero, and any above max Z level.
static void filterPoints(std::vector<float3>& out, const std::vector<LasPoint>& in) {
  out.reserve(in.size());
  for (int i=0; i<in.size(); i++) {
    LasPoint p = in[i];
    if (p.z >= 0 and p.z < 1) {
      out.push_back(make_float3(p.x,p.y,p.z));
    }
  }
}


static __global__ void is_cell_bad_(bool* out, const int64_t* pts, const int32_t* cnts, int C) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < C) {
    out[i] = cnts[i] < 6;
  }
}
static __global__ void assign_pt_good_or_bad(bool* stencil,
    const float* pts1, int N, int lvl, float scale,
    const bool* cellIsBad, const int64_t* lvlInds, int C) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  bool good = false;
  //int x_off=0, y_off=0, z_off=0;
  for (int x_off=-1; x_off<2; x_off++)
  for (int y_off=-1; y_off<2; y_off++)
  for (int z_off=0; z_off<1; z_off++) {
    int x = pts1[i*3+0] * scale + x_off;
    int y = pts1[i*3+1] * scale + y_off;
    int z = pts1[i*3+2] * scale + z_off;
    int idx = binSearch4_encoded(lvl, x,y,z, lvlInds, C);
    if (idx >= 0 and idx < C) good |= not cellIsBad[idx];
  }
  stencil[i] = good;
}
// Swap these to see the deleted points.
struct is_true { __host__ __device__ bool operator()(const bool& b) { return b == true; } };
//struct is_true { __host__ __device__ bool operator()(const bool& b) { return b == false; } };
int filter_outliers(
    float3* outPts,
    float3* pts1, int N,
    int inspectLvl,
    const SortedOctree<int32_t>& oct) {
  int C = oct.lvlSizes[inspectLvl];
  const int64_t* lvlInds = oct.inds+oct.lvlOffsets[inspectLvl];
  const int32_t* lvlCnts = oct.vals+oct.lvlOffsets[inspectLvl];
  bool* cellIsBad = 0, *stencil=0;
  cudaMalloc(&cellIsBad, C); cudaMalloc(&stencil, N);

  // Determine which cells are bad.
  is_cell_bad_<<<(C+127)/128,128>>>(cellIsBad, lvlInds, lvlCnts, C);
  //cudaDeviceSynchronize(); getLastCudaError("post is_cell_bad"); printf(" - [hmr] assiging good / bad.\n");

  // Assign every pt to a cell, then copy only those that aren't bad.
  float scale = 1 << inspectLvl;
  assign_pt_good_or_bad<<<(N+127)/128,128>>>(stencil, (float*)pts1, N, inspectLvl, scale, cellIsBad, lvlInds, C);
  //cudaDeviceSynchronize(); getLastCudaError("post assign_pt"); printf(" - [hmr] copying.\n");
  int n = thrust::copy_if(thrust::device, pts1, pts1+N, stencil, outPts, is_true()) - outPts;
  cudaFree(cellIsBad); cudaFree(stencil);
  cudaDeviceSynchronize(); getLastCudaError("post copy_if"); printf(" - [hmr] finished filtering outliers.\n");
  return n;
}

/*
static __global__ void rasterize_(float2 *img, int w, int h, float3* pts, int N) {
  int y = blockIdx.x, x = blockIdx.y;
  int j = threadIdx.x;
  //int A = (N + (blockDim.x - 1)) / blockDim.x;
  int A = 1024;
  float weight = 0;
  float max = 0;
  // Whats better in terms of threads in a block: strided or planar -by-thread access?
  if (j == 0) printf(" - starting block %d %d\n", y,x);
  for (int pi=j; pi<N; pi+=A) {
    float3 pt = pts[pi];
    int px = pt.x * w + .5;
    int py = pt.y * h + .5;
    float pz = pt.z;
    if (px == x and py == y) {
      if (pz > max or weight == 0) {
        max = pz;
        weight = 1;
      }
    }
  }
  __shared__ float2 tmp[1024];
  tmp[j] = make_float2(max, weight);
  for (int k=512; k>1; k>>=1) {
    __syncthreads();
    if (j < k and tmp[j+k].y > 0 and (tmp[j+k].x > tmp[j].x or tmp[j].y == 0)) tmp[j] = tmp[j+k];
  }
  __syncthreads();
  if (j == 0) {
    img[y*w+x] = tmp[0];
  }
}
*/
// Reduce max, by pixel
static __global__ void fill_vals(float2* vals, float* vals0, int N) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < N) vals[i] = make_float2(vals0[i], 1.);
  //if (i < N) vals[i] = make_float2(vals0[i], .4);
}
void orthographically_rasterize(CuImage<float2>& out, float* pts, int N) {
  //dim3 blk(out.h, out.w);
  //dim3 thr(1024);
  //rasterize_<<<blk,thr>>>(out.buf, out.w, out.h, (float3*)pts, N);

  // Transform to get keys/vals
  // Sort keys
  // Reduce max
  // Scatter to image output
  int *keys = 0; cudaMalloc(&keys, sizeof(int)*2*N);
  float *vals = 0; cudaMalloc(&vals, sizeof(float)*1*N);
  int *keys2 = 0; cudaMalloc(&keys2, sizeof(int)*2*N);
  float *vals2 = 0; cudaMalloc(&vals2, sizeof(float)*2*N);
  float2 *vals3 = 0; cudaMalloc(&vals3, sizeof(float2)*1*N);
  int w = out.w, h = out.h;
  thrust::transform(thrust::device, (float3*)pts, ((float3*)pts)+N, keys,
        //[=] __device__ (const float3& pt) { return min((int)(pt.x*w+.5f),w) + w*min(h,(int)(pt.y*h+.5f)); });
        //[=] __device__ (const float3& pt) { return (int)(pt.x*w+.5f) + w*(int)(pt.y*h+.5f); });
        [=] __device__ (const float3& pt) {
          int i = max(0,min((int)(pt.x*w+.5f),w-1)) + w*max(0,min(h-1,(int)(pt.y*h+.5f)));
          //if (i<0 or i>=w*h) printf(" - %d / %d\n", i, w*h);
          return i; });
  thrust::transform(thrust::device, (float3*)pts, ((float3*)pts)+N, vals,
        [=] __device__ (const float3& pt) { return pt.z; });
  thrust::sort_by_key(thrust::device, keys, keys+N, vals);
  thrust::sort(thrust::device, keys, keys+N);
  //int N2 = thrust::reduce_by_key(thrust::device, keys,keys+N, vals, keys2,vals2).first - keys2;
  int N2 = thrust::reduce_by_key(thrust::device, keys,keys+N, vals, keys2,vals2,
      thrust::equal_to<int>(),
      //[]__device__(const float& a, const float &b) { return (a+b)*.5; }).first - keys2;
      []__device__(const float& a, const float &b) { return a>b?a:b; }).first - keys2;
  //scatter_<<<(
  fill_vals<<<(N2+127)/128, 128>>>(vals3, vals2, N2);
  thrust::fill(thrust::device, out.buf, out.buf+out.w*out.h, make_float2(0,0));
  thrust::scatter(thrust::device, vals3, vals3+N2, keys2, (float2*)out.buf);
  cudaDeviceSynchronize(); getLastCudaError("post rasterize_");
  cudaFree(keys);
  cudaFree(keys2);
  cudaFree(vals);
  cudaFree(vals2);
  cudaFree(vals3);
}

// Fill in empty gaps by pyramidal push-pull
static __device__ float clamp(const float& a, float l, float h) { return a < l ? l : a > h ? h : a; }
static __global__ void filter_lvl_down(float2* dst, int dw, int dh, const float2* src, int sw, int sh, bool isLast) {
  // Todo optimize by shared mem fetching
  int tx = threadIdx.y + blockIdx.y * blockDim.y;
  int ty = threadIdx.x + blockIdx.x * blockDim.x;

  if (tx<dw and ty<dh) {

    int sy = 2 * ty;
    int sx = 2 * tx;

    constexpr float weights_[16] = {
      1/64., 3./64, 3./64, 1./64,
      3/64., 9./64, 9./64, 3./64,
      3/64., 9./64, 9./64, 3./64,
      1/64., 3./64, 3./64, 1./64 };

    float2 here = make_float2(0,0);

    for (int ky=-1; ky<3; ky++)
    for (int kx=-1; kx<3; kx++)
      if (sy+ky < sh and sx+kx < sw and sy+ky >= 0 and sx+kx >= 0) {
        float2 a = src[(sy+ky)*sw + sx+kx];
        here.x += a.x * weights_[(ky+1)*4+kx+1];
        here.y += a.y * weights_[(ky+1)*4+kx+1];
    }
    float w = clamp(here.y, .0000001f, 999.f);
    //float w = clamp(here.y, .0f, 999.f);
    float wc = 1. - pow((1.f - clamp(w,0.f,1.f)), 16.f);
    //float wc = 1. - pow((1.f - clamp(w,0.f,1.f)), 4.f);
    float h = (here.x / w) * (isLast ? 1 : wc);
    dst[ty*dw+tx] = make_float2(h,wc);
  }
}
static __global__ void filter_lvl_up(float2* dst, int dw, int dh,
    const float2* same,
    const float2* next, int nw, int nh
    ) {
  int tx = threadIdx.y + blockIdx.y * blockDim.y;
  int ty = threadIdx.x + blockIdx.x * blockDim.x;

  if (tx<dw and ty<dh) {

    int ny = ty >> 1;
    int nx = tx >> 1;

    constexpr float weights_[16] = {
      1/64., 3./64, 3./64, 1./64,
      3/64., 9./64, 9./64, 3./64,
      3/64., 9./64, 9./64, 3./64,
      1/64., 3./64, 3./64, 1./64 };

    float2 here = make_float2(0,0);

    for (int ky=-1; ky<3; ky++)
    for (int kx=-1; kx<3; kx++)
      if (ny+ky < nh and nx+kx < nw and ny+ky >= 0 and nx+kx >= 0) {
        float2 a = next[(ny+ky)*nw + nx+kx];
        here.x += a.x * weights_[(ky+1)*4+kx+1];
        here.y += a.y * weights_[(ky+1)*4+kx+1];
    }

    float2 a0 = same[ty*dw+tx];

    float wc = a0.y;
    //float newx = (1. - wc) * here.x + a0.x;
    float newx = (1. - wc) * here.x + wc * a0.x;

    // actually y coord / weight not used anymore
    dst[ty*dw+tx] = make_float2(newx, here.y);
  }
}
template <int N>
static __global__ void median_filter_(float* out, const float* in, int w, int h) {
  int ty = threadIdx.x + blockIdx.x * blockDim.x;
  int tx = threadIdx.y + blockIdx.y * blockDim.y;
  //int gy = blockIdx.x;
  //int gx = blockIdx.y;

  int n = 0;
  float grp[9];
  constexpr int m = N / 2;
  constexpr int NN = N * N;
  for (int dy=-m; dy<=m; dy++)
  for (int dx=-m; dx<=m; dx++)
    if (ty+dy >= 0 and ty+dy < h and tx+dx >= 0 and tx+dx < w) {
      grp[n++] = in[(ty+dy)*w+tx+dx];
    }

  for (int i=0; i<NN; i++) {
    int best = i;
    for (int j=i+1; j<NN; j++) {
      if (grp[j] < grp[best]) {
        best = j;
      }
    }
    float tmp = grp[i];
    grp[i] = grp[best];
    grp[best] = tmp;
  }
  out[ty*w+tx] = grp[(N*N+1)/2];
}

void push_pull_filter(CuImage<float>& out, CuImage<float2>& base) {
  constexpr int nlvls = 4;
  CuImage<float> outRaw;
  CuImage<float2> lvls1[nlvls];

  int W = base.w, H = base.h;
  int w=W, h=H;

  CuImage<float2>* lastLvl = &base;

  CuImage<float2> passed[2];
  passed[0].allocate(W,H,1);
  passed[1].allocate(W,H,1);

  printf(" - [hmr] push-pull filtering.\n");

  lvls1[0].allocate(w,h,1);
  cudaMemcpy(lvls1[0].buf, base.buf, sizeof(float2)*w*h, cudaMemcpyDeviceToDevice);
  //show_img(lvls1[0], "down", 0);

  //show_img(base, "base", 0);
  for (int i=1; i<nlvls; i++) {
    w>>=1; h>>=1;
    lvls1[i].allocate(w,h,1);
    //lvls1[i].allocate(2*w,2*h,1);

    dim3 blk((h+15)/16, (w+15)/16);
    dim3 thr(16,16);
    filter_lvl_down<<<blk,thr>>>(
        lvls1[i].buf, w,h,
        lastLvl->buf, lastLvl->w, lastLvl->h, i==nlvls);
    lastLvl = &lvls1[i];
    cudaDeviceSynchronize();

    //show_img(lvls1[i], "down", 0);
  }

  w<<=1; h<<=1;
  for (int i=nlvls-2; i>=0; i--) {

    dim3 blk((h+15)/16, (w+15)/16);
    dim3 thr(16,16);
    filter_lvl_up<<<blk,thr>>>(
        passed[i%2].buf, w,h,
        lvls1[i].buf,
        i == nlvls-2 ? lvls1[i+1].buf : passed[(i+1)%2].buf, lvls1[i+1].w, lvls1[i+1].h);
    lastLvl = &lvls1[i];
    cudaDeviceSynchronize();

    //show_img(passed[i%2], "up", 0, w,h);
    w<<=1; h<<=1;
  }
  cudaDeviceSynchronize(); getLastCudaError("post push-pull");

  outRaw.allocate(W,H,1);

  // Copy output xy -> x
  thrust::transform(thrust::device, passed[0].buf, passed[0].buf+W*H, outRaw.buf, []__device__(const float2& f) { return f.x; });
  cudaDeviceSynchronize(); getLastCudaError("post transform");

  // Free old buffers
  for (int i=0; i<nlvls; i++) lvls1[i].release();
  for (int i=0; i<2; i++) passed[i].release();
  cudaDeviceSynchronize(); getLastCudaError("post release");

  // Do median filter
  out.allocate(W,H,1);
  dim3 blk((H+15)/16, (W+15)/16);
  dim3 thr(16,16);
  cudaDeviceSynchronize(); getLastCudaError("pre-median");
  //median_filter_<3><<<blk,thr>>>(out.buf, outRaw.buf, W,H);
  median_filter_<5><<<blk,thr>>>(out.buf, outRaw.buf, W,H);
  //show_img(out, "filtered", 0, W,H);
  cudaDeviceSynchronize(); getLastCudaError("post-median");
  outRaw.release();
  cudaDeviceSynchronize(); getLastCudaError("post-free");
}

const int8_t PLUS_Z = 1;
const int8_t MNUS_Z = 2;
const int8_t PLUS_X = 4;
const int8_t MNUS_X = 8;
const int8_t PLUS_Y = 16;
const int8_t MNUS_Y = 32;
static __global__ void fillLvl(
    int4* inds, int G, int lvl,
    float* surface) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (y<0 or y>=G or x<0 or x>=G) return;

  float factor = 1. / ((float)G);
  float myZ = ((float)lvl) * factor;
  float mySurfaceZ = surface[y*G+x];

  int8_t dirMask = 0;

  float myLastZ = ((float)(lvl-1)) * factor;
  // nope
  if (myLastZ > mySurfaceZ) return;
  if (myZ > mySurfaceZ) dirMask = PLUS_Z;

  for (int dd=0; dd<4; dd++) {
    int dx = dd == 0 ? 1 : dd == 1 ? -1 : 0;
    int dy = dd == 2 ? 1 : dd == 3 ? -1 : 0;

    int yy = y+dy;
    int xx = x+dx;
    if (yy>=0 && yy < G && xx>=0 && xx < G) {
      float neighborSurfaceZ = surface[yy*G+xx];
      //if (myZ > neighborSurfaceZ) dirMask |= ((i!=0)*4*(i+3)) | ((j!=0)*2*(j+3));
      if (myZ > neighborSurfaceZ and neighborSurfaceZ < myLastZ)
        if (dx ==  1) dirMask |= PLUS_X;
        else if (dx == -1) dirMask |= MNUS_X;
        else if (dy ==  1) dirMask |= PLUS_Y;
        else dirMask |= MNUS_Y;
    }
  }

  int i = y*G+x;
  if (dirMask) {
    //printf(" %d | %d %d %d\n", i,x,y,lvl);
    //printf(" %d | %d %d %d %f\n", i,x,y,lvl,mySurfaceZ);
    inds[i] = make_int4(x,y,lvl,dirMask);
  } else {
    inds[i] = make_int4(-1,-1,-1,-1);
  }
}
int build_initial_surface(int4*& outVerts, const CuImage<float>& surface) {
  cudaDeviceSynchronize(); getLastCudaError("pre build surface");
  int G = surface.w;
  float surface_max = thrust::reduce(thrust::device, surface.buf, surface.buf+surface.w*surface.h, 0.f, thrust::maximum<float>());
  float surface_min = thrust::reduce(thrust::device, surface.buf, surface.buf+surface.w*surface.h, 0.f, thrust::minimum<float>());
  printf(" - min max surface: %f %f\n", surface_min, surface_max);
  //float surface_max = .1;
  //float surface_min = 0;
  int max_z_lvl = surface_max * surface.w + 1;
  int min_z_lvl = surface_min * surface.w;

  int4* lvlVerts, *lvlVerts2;
  int N = G*G;
  cudaMalloc(&lvlVerts, sizeof(int4)*N);
  cudaMalloc(&lvlVerts2, sizeof(int4)*N);

  thrust::device_vector<int4> verts0_;
  auto verts0 = &verts0_;
  int nn = 0;

  for (int l=min_z_lvl; l<max_z_lvl; l++) {
    thrust::fill(thrust::device, lvlVerts, lvlVerts+N, make_int4(-1,-1,-1,-1));
    cudaDeviceSynchronize(); getLastCudaError("post fill1");
    dim3 blk((G+15)/16, (G+15)/16);
    dim3 thr(16,16);
    fillLvl<<<blk,thr>>>(lvlVerts, G,l, surface.buf);
    cudaDeviceSynchronize(); getLastCudaError("post fill");

    int n_old = nn;
    auto it = thrust::copy_if(thrust::device, lvlVerts, lvlVerts+N, lvlVerts2,
        []__device__(const int4 &a) { return a.x != -1; });
    cudaDeviceSynchronize(); getLastCudaError("post copy_if");
    int n_lvl = it - lvlVerts2;

    verts0->resize(n_old + n_lvl);
    cudaDeviceSynchronize(); getLastCudaError("post resize");
    cudaMemcpy(
        ((int4*)thrust::raw_pointer_cast(verts0->data())) + n_old,
        lvlVerts2, sizeof(int4)*n_lvl, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize(); getLastCudaError("post copy");
    nn = verts0->size();
    nn += n_lvl;
  }

  cudaMalloc(&outVerts, sizeof(int4)*nn);
  cudaMemcpy(outVerts, thrust::raw_pointer_cast(verts0->data()), sizeof(int4)*nn, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize(); getLastCudaError("post final memcpy");
  return nn;
}

void HeightMapRasterizer::run(const std::vector<LasPoint>& pts0) {
  // Filter points.
  std::vector<float3> pts1;
  filterPoints(pts1, pts0);
  printf(" - [hmr] filtered, have %d / %d\n", pts1.size(), pts0.size());

  //int3* ptsq;
  //cudaMalloc(&ptsq, sizeof(int3)*pts1.size());
  //binPoints(ptsq, pts1, resolution);

  int N1 = pts1.size();
  float *dev_pts1 = 0;
  cudaMalloc(&dev_pts1, sizeof(float)*3*N1);
  cudaMemcpy(dev_pts1, pts1.data(), sizeof(float)*3*N1, cudaMemcpyHostToDevice);
  printf(" - [hmr] copied to device.\n");

  //SortedOctree<void> tree;
  SortedOctree<int32_t> tree;
  int32_t* vals = (int32_t*) malloc(sizeof(int32_t)*N1);
  for (int i=0; i<N1; i++) vals[i] = 1;
  //createOctree<void>(tree, (float3*)dev_pts1, N1, 16, 5);
  tree.createOctree((float*)dev_pts1, vals, N1, 16, 5);

  // To delete some outliers, look at the maximum Z jump in neighborhoods in the XY plane.
  // NO
  // Just look for very low density regions at some depth lower than the maximum.
  // If a cell is disconnected, delete it and all of it's children.
  float* dev_pts2 = 0; cudaMalloc(&dev_pts2, sizeof(float)*3*N1);
  printf(" - [HeightMapRasterizer] calling filter_outliers().\n");
  int N2 = filter_outliers((float3*)dev_pts2, (float3*)dev_pts1, N1, 7, tree);
  printf(" - Have %d / %d inlying.\n", N2, N1);
  inlyingPoints.resize(3*N2);
  cudaMemcpy(inlyingPoints.data(), dev_pts2, sizeof(float)*N2*3, cudaMemcpyDeviceToHost);
  cudaFree(dev_pts1);

  // Build tree again, this time without outliers
  // Note I could write some kernels to operate on nodes directly, but this is easer
  //createOctree(tree, (float3*)dev_pts2, N2, 16, 5);

  // Create raw heightmap
  CuImage<float2> map; // first channel is value, second is weight
  CuImage<float> finalMap;
  //int mapRes = 2048;
  int mapRes = 1024;
  map.allocate(mapRes,mapRes,1);
  finalMap.allocate(mapRes,mapRes,1);
  printf(" - [HeightMapRasterizer] calling orthographically_rasterize().\n");
  orthographically_rasterize(map, dev_pts2, N2);
  printf(" - [hmr] done rasterizing.\n");

  /*
  cv::Mat dimg(mapRes,mapRes, CV_32F);
  cudaMemcpy2D(
      dimg.data, sizeof(float)*mapRes,
      map.buf, sizeof(float)*mapRes*2, mapRes,mapRes, cudaMemcpyDeviceToHost);
  cv::imshow("dimg", dimg); cv::waitKey(0);
  */

  cudaFree(dev_pts2);

  // Push-Pull filter to fill in empty gaps
  push_pull_filter(finalMap, map);
  cudaDeviceSynchronize(); getLastCudaError("post push pull filter");

  // Build initial surface
  printf(" - [HeightMapRasterizer] calling build_initial_surface().\n");
  int4* verts;
  int ns = build_initial_surface(verts, finalMap);

#if 0
  int4* vertsCpu = (int4*)malloc(sizeof(int4)*ns);
  cudaMemcpy(vertsCpu, verts, sizeof(int4)*ns, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  inlyingPoints.resize(ns*3);
  for (int i=0; i<ns; i++) {
    int4 &p = vertsCpu[i];
    inlyingPoints[i*3+0] = p.x / ((float)mapRes);
    inlyingPoints[i*3+1] = p.y / ((float)mapRes);
    inlyingPoints[i*3+2] = p.z / ((float)mapRes);
    //if (i % 50000 == 0) printf(" - %d | %d %d %d\n", i, vertsCpu[i].x,vertsCpu[i].y,vertsCpu[i].z);
  }
  free(vertsCpu);
#endif

  // Build quads
  GpuBuffer<int3> tris;
  GpuBuffer<int4> quads;
  GpuBuffer<int> vert2tris;
  int G = mapRes;
  printf(" - [HeightMapRasterizer] calling makeGeo().\n");
  makeGeo(meshVerts, meshTris, tris, quads, vert2tris, verts, ns, G);

  // Fin
  cudaFree(verts);
}
