#include "controllable_camera.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

ControllabeCamera::ControllabeCamera(const CamSpec& cs)
  : Camera(cs)
{
}

void ControllabeCamera::setPos(const double t[3]) {
  this->t[0] = t[0];
  this->t[1] = t[1];
  this->t[2] = t[2];
}
void ControllabeCamera::setRot(const double R_[9]) {
  Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> R(R_);
  Eigen::Quaterniond qq { R };
  q[0] = qq.w();
  q[1] = qq.x();
  q[2] = qq.y();
  q[3] = qq.z();
}
void ControllabeCamera::getPos(double t[3]) {
  t[0] = this->t[0];
  t[1] = this->t[1];
  t[2] = this->t[2];
}
void ControllabeCamera::getRot(double R[9]) {
  Eigen::Quaterniond qq { q[0], q[1], q[2], q[3] };
  Eigen::Matrix<double,3,3,Eigen::RowMajor> R_ = qq.toRotationMatrix();
  for (int i=0; i<3; i++)
  for (int j=0; j<3; j++)
    R[i*3+j] = R_(i,j);
}

void ControllabeCamera::step(double dt) {
  Eigen::Quaterniond qq { q[0], q[1], q[2], q[3] };
  double aspeed = .3;
  double dy = curY - lastY, dx = curX - lastX;
  if (not leftMouseDown) dy = dx = 0;
  if (lastY == -1 or dy > 100 or dy < -100 or dx > 100 or dx < -100) dy = dx = 0;

  Eigen::Quaterniond dq = Eigen::Quaterniond {
    (Eigen::AngleAxisd(dt*aspeed*dy, Eigen::Vector3d::UnitX()) *
     Eigen::AngleAxisd(dt*aspeed*dx, Eigen::Vector3d::UnitY())) };
  Eigen::Quaterniond qq2 = dq * qq;

  Eigen::Matrix<double,3,3,Eigen::RowMajor> R = qq2.toRotationMatrix();
  q[0] = qq2.w(); q[1] = qq2.x(); q[2] = qq2.y(); q[3] = qq2.z();

  Eigen::Map<Eigen::Vector3d> t_(t);
  Eigen::Map<Eigen::Vector3d> vel_(vel);

  double speed = 25;
  Eigen::Vector3d accRaw {
    (keyDown['a'-'a'] ?  1 : keyDown['d'-'a'] ? -1 : 0) * speed,
    0,
    (keyDown['w'-'a'] ?  1 : keyDown['s'-'a'] ? -1 : 0) * speed };
  Eigen::Vector3d acc = R.transpose() * accRaw;
  //acc += -(vel_.array()*vel_.cwiseAbs().array()).matrix();
  acc += -vel_;
  vel_ += acc * dt;
  t_ += vel_ * dt + acc * dt * .5;

  Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor>> V(view);
  V.topLeftCorner<3,3>() = R;
  V.topRightCorner<3,1>() = -R * t_;
  //std::cout << " - view:\n" << V << "\n";

  Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor>> P(proj);
  //std::cout << " - proj:\n" << P << "\n";

  lastX = curX; lastY = curY;
}

void ControllabeCamera::keyboardFunc(int key, int scancode, int action, int mods) {
  int c = key - GLFW_KEY_A;
  if (action == GLFW_PRESS) {
    if (c >= 0 and c < 26) keyDown[c] = true;
    if (key == GLFW_KEY_SPACE) spaceDown = true;
    if (key == GLFW_KEY_ESCAPE) escDown = true;
  } else if (action == GLFW_RELEASE) {
    if (c >= 0 and c < 26) keyDown[c] = false;
    if (key == GLFW_KEY_SPACE) spaceDown = false;
    if (key == GLFW_KEY_ESCAPE) escDown = false;
  }
  shiftDown = mods & GLFW_MOD_SHIFT;
  ctrlDown = mods & GLFW_MOD_CONTROL;
}
void ControllabeCamera::clickFunc(int button, int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) leftMouseDown = action ==  GLFW_PRESS;
  if (button == GLFW_MOUSE_BUTTON_RIGHT) rightMouseDown = action == GLFW_PRESS;
}
void ControllabeCamera::motionFunc(double xpos, double ypos) {
  lastX = curX, lastY = curY;
  curY = ypos, curX = xpos;
}

