
#include "camera.h"
#include <GL/glew.h>
#include <iostream>

Camera1::Camera1(int w, int h, float fov)
  : w(w), h(h), fov(fov)
{
  float n = .01;
  float f = 10.;
  u = (((float)w)/h) * std::tan(fov/2.);
  v = 1. * std::tan(fov/2.);
  float l = -n * u;
  float r = n * u;
  float t = n * v;
  float b = -n * v;
  proj <<
    2*n/(r-l), 0, 0, 0,
    0, 2*n/(t-b), 0, 0,
    0, 0, -(f+n)/(f-n), -2*f*n/(f-n),
    0, 0, -1, 0;
  //std::cout << " - Proj:\n" << proj << "\n";

  view.setIdentity();
  view(2,3) = -1;

  curX=lastX=curY=lastY=-1;
  q.setIdentity();
  this->t = Eigen::Vector3f{0,0,1.f};
}

void Camera1::bind() {
  glMatrixMode(GL_PROJECTION);
  glLoadMatrixf(proj.data());

  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixf(view.data());
}
void Camera1::unbind() {
}
void Camera1::update(float dt) {

  if (leftDown and
      not (lastX == -1 or lastY == -1)) {
    Eigen::Quaternionf dq =
      Eigen::AngleAxisf(.2*dt*(curX-lastX), Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(.2*dt*(curY-lastY), Eigen::Vector3f::UnitX());
    q = q * dq;
  }

  view.topLeftCorner<3,3>() = q.matrix().transpose();
  view.topRightCorner<3,1>() = -q.matrix().transpose() * t;

  Eigen::Vector3f dtrans {
    keyDown['A'-GLFW_KEY_A] ? dt : keyDown['D'-GLFW_KEY_A] ? -dt : 0,
    keyDown['Q'-GLFW_KEY_A] ? dt : keyDown['E'-GLFW_KEY_A] ? -dt : 0,
    keyDown['W'-GLFW_KEY_A] ? dt : keyDown['S'-GLFW_KEY_A] ? -dt : 0 };
  t = t + q.matrix() * dtrans * -.2;

  //std::cout << " - " << curX << " " << curY << " : " << lastX << " : " << lastY << "\n";
  //std::cout << " - View:\n" << view << "\n";
  lastX = curX; lastY = curY;
}

void Camera1::reshapeFunc(int w, int h) {
}
void Camera1::keyboardFunc(int key, int scancode, int action, int mods) {
  keyDown[key-GLFW_KEY_A] = action != GLFW_RELEASE;
  if (key == GLFW_KEY_Q) wantToQuit = true;
}
void Camera1::clickFunc(int button, int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT and action == GLFW_PRESS) leftDown = true;
  if (button == GLFW_MOUSE_BUTTON_LEFT and action == GLFW_RELEASE) leftDown = false;
  if (button == GLFW_MOUSE_BUTTON_RIGHT and action == GLFW_PRESS) rightDown = true;
  if (button == GLFW_MOUSE_BUTTON_RIGHT and action == GLFW_RELEASE) rightDown = false;
}
void Camera1::motionFunc(double xpos, double ypos) {
  curX = xpos; curY = ypos;
  if (lastX == -1) lastX = xpos, lastY = ypos;
}
