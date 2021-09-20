#include "math.h"
#include <Eigen/LU>

void mat44_double_to_float_invt(float C_[16], const double A_[16]) {
  Eigen::Map<Eigen::Matrix<float,4,4,Eigen::RowMajor>> C(C_);
  Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::RowMajor>> A(A_);
  C = A.inverse().transpose().cast<float>();
}
