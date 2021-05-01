#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "window.h"

struct Camera1 : public UsesIO {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Camera1(int w, int h, float fov);

  Eigen::Matrix<float, 4,4> proj;
  Eigen::Matrix<float, 4,4> view;
  Eigen::Quaternionf q;
  Eigen::Vector3f t;

  void bind();
  void unbind();
  void update(float dt);

  int w, h;
  float fov, u, v;

  virtual void reshapeFunc(int w, int h) override;
  virtual void keyboardFunc(int key, int scancode, int action, int mods) override;
  virtual void clickFunc(int button, int action, int mods) override;
  virtual void motionFunc(double xpos, double ypos) override;

  double lastX=-1, lastY=-1, curX=-1, curY=-1;
  bool leftDown=false, rightDown=false;
  bool keyDown[26] = {false};

};
