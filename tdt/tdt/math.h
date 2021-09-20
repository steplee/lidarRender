#pragma once

// TODO: SIMD (AVX256 / NEON)

// row-major
// NOT OKAY FOR IN-PLACE OPERATION (A != C != B)
inline void matmul44(double C[16], const double A[16], const double B[16]) {
  for (int i=0; i<4; i++)
  for (int j=0; j<4; j++) {
    double sum = 0;
    for (int k=0; k<4; k++)
      sum += A[i*4+k] * B[k*4+j];
    C[i*4+j] = sum;
  }
}
inline void matmul44_colMajor(double C[16], const double A[16], const double B[16]) {
  for (int i=0; i<4; i++)
  for (int j=0; j<4; j++) {
    double sum = 0;
    for (int k=0; k<4; k++)
      sum += A[k*4+i] * B[j*4+k];
    C[i*4+j] = sum;
  }
}

inline void matmul44_double_to_float(float C[16], const double A[16], const double B[16]) {
  for (int i=0; i<4; i++)
  for (int j=0; j<4; j++) {
    double sum = 0;
    for (int k=0; k<4; k++)
      sum += A[i*4+k] * B[k*4+j];
    C[i*4+j] = (float) sum;
  }
}

void mat44_double_to_float_invt(float C[16], const double A[16]);
