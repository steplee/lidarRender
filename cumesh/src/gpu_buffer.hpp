

template <typename T> struct GpuBuffer {
  int allocatedCnt = 0;
  T* buf = 0;

  inline GpuBuffer() {}
  // No copy constructor
  GpuBuffer(const GpuBuffer& a) = delete;

  inline void allocate(int n) {
    if (allocatedCnt < n) {
      if (buf) cudaFree(buf);
      cudaMallocManaged(&buf, sizeof(T)*n);
      allocatedCnt = n;
    }
  }

};
