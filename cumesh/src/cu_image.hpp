#pragma once

#include <cuda_runtime.h>

template <class T>
struct CuImage {
  T* buf = 0;
  int allocatedSize = 0;
  int w, h, c, pitch;
  void allocate(int ww, int hh, int cc) {
    w=ww; h=hh; c=cc;
    allocatedSize = w*h*c;
    if (buf) cudaFree(buf);
    cudaMalloc(&buf, allocatedSize * sizeof(T));
    pitch = w*sizeof(float);
  }
  void release() {
    if (buf != 0) cudaFree(buf);
    w=h=c=pitch=allocatedSize = 0;
    buf = 0;
  }
  ~CuImage() { release(); }
  inline CuImage() {};
  CuImage(const CuImage& other) = delete;
  CuImage operator=(const CuImage& other) = delete;
};
