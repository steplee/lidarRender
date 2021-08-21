#pragma once

struct CamSpec {

  CamSpec(float hfov, float vfov, float aspect);

  float hfov, vfov;
  float u, v;
  float aspect;
  float near, far;

  void proj(double out[16]);
};

class Camera {
  public:
    Camera(const CamSpec& s);

    void viewProj(double out[16]);

    void updateView(const double in[16]);
    void updateCamSpec(const CamSpec& s);
    void updateNearFar(float n, float f);

  private:
    CamSpec camSpec;

    double view[16];
    double proj[16];
};
