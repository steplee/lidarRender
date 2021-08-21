#include "camera.h"
#include <math.h>
#include <cstring>

CamSpec::CamSpec(float hfov, float vfov, float aspect) {
  this->hfov = hfov;
  this->vfov = vfov;
  this->aspect = aspect;
  u = .5 / std::tan(hfov * .5 * M_PI / 180.);
  v = .5 / std::tan(vfov * .5 * M_PI / 180.);
  near = .1;
  far = 100;
}

void CamSpec::proj(double out[16]) {
  double out_[] = {
    near / u, 0, 0, 0,
    0, near / v, 0, 0,
    0, 0, (far+near) / (far-near), -2*far*near / (far - near),
    0, 0, 1, 0
  };
  memcpy(out, out_, sizeof(double)*16);
}






Camera::Camera(const CamSpec& s) : camSpec(s) {
  for (int i=0; i<4; i++) view[i] = i % 5 == 0;
  camSpec.proj(proj);
}

void Camera::updateView(const double in[16]) {
  memcpy(view, in, sizeof(double)*16);
}
void Camera::updateCamSpec(const CamSpec& s) {
  camSpec = s;
  camSpec.proj(proj);
}
void Camera::updateNearFar(float n, float f) {
  camSpec.near = n;
  camSpec.far  = f;
  camSpec.proj(proj);
}
