#include "camera.h"
#include <math.h>
#include <cstring>
#include <iostream>

#include <Eigen/Core>

CamSpec::CamSpec(float hfov, float vfov, float aspect) {
  this->hfov = hfov;
  this->vfov = vfov;
  this->aspect = aspect;
  u = 1 / std::tan(hfov * .5 * M_PI / 180.);
  v = 1 / std::tan(vfov * .5 * M_PI / 180.);
  near = .2;
  far = 500;
}

void CamSpec::proj(double out[16]) {
  double out_[] = {
    u, 0, 0, 0,
    0, v, 0, 0,
    //0, 0, (far+near) / (far-near), -2*far*near / (far - near),
    //0, 0, 1, 0
    0, 0,  (far+near) / (far-near),  -2*far*near / (far - near),
    0, 0,  1, 0
  };
  memcpy(out, out_, sizeof(double)*16);
}


void Camera::setPos(const double t_[3]) {
  Eigen::Matrix<double,3,1> t { t_[0], t_[1], t_[2] };
  Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor>> V(view);
  V.topRightCorner<3,1>() = -V.topLeftCorner<3,3>() * t;
}
void Camera::setRot(const double R_[9]) {
  Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> R(R_);
  Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor>> V(view);
  Eigen::Matrix<double,3,1> t = -V.topLeftCorner<3,3>().transpose() * V.topRightCorner<3,1>();
  V.topLeftCorner<3,3>() = R;
  V.topRightCorner<3,1>() = -R * t;
}
void Camera::getPos(double t[3]) {
  Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor>> V(view);
  Eigen::Vector3d tt = V.topLeftCorner<3,3>().transpose() * -V.topRightCorner<3,1>();
  t[0] = tt(0);
  t[1] = tt(1);
  t[2] = tt(2);
}
void Camera::getRot(double R[9]) {
  for (int i=0; i<3; i++)
  for (int j=0; j<3; j++)
    R[i*3+j] = view[j*4+i];
}






Camera::Camera(const CamSpec& s) : camSpec(s) {
  for (int i=0; i<16; i++) view[i] = i % 5 == 0;
  camSpec.proj(proj);
}

//void Camera::updateView(const double in[16]) { memcpy(view, in, sizeof(double)*16); }

void Camera::updateCamSpec(const CamSpec& s) {
  camSpec = s;
  camSpec.proj(proj);
}
void Camera::updateNearFar(float n, float f) {
  camSpec.near = n;
  camSpec.far  = f;
  camSpec.proj(proj);
}

void Camera::viewProj(double out[16]) {
  Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor>> P(proj);
  Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor>> V(view);
  Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor>> PV(out);
  PV = P * V;
  //std::cout << " - mvp matrix:\n" << PV << "\n";
}

bool Camera::isSame(const Camera& other) {
  for (int i=0; i<16; i++) if (other.view[i] != view[i]) return false;
  for (int i=0; i<16; i++) if (other.proj[i] != view[i]) return false;
  return true;
}
