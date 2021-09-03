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

    void updateCamSpec(const CamSpec& s);
    void updateNearFar(float n, float f);

    virtual void step(double dt) {};
    bool isSame(const Camera& other);

    virtual void setPos(const double t[3]);
    virtual void setRot(const double R[9]);
    virtual void getPos(double t[3]);
    virtual void getRot(double R[9]);
    inline float getU() { return camSpec.u; }
    inline float getV() { return camSpec.v; }

    inline const CamSpec& getSpec() { return camSpec; }

  protected:
    CamSpec camSpec;

    alignas(8) double view[16];
    alignas(8) double proj[16];

};

