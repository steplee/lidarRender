#include "viz_app.h"

VizApp::VizApp(int w, int h) : w(w), h(h) {
}

void VizApp::startOnThisThread() {
  init();
}
void VizApp::startOnOwnThread() {
  haveThread = true;
  internalThread = std::thread(&VizApp::internalLoop, this);
}

void VizApp::init() {
  window = new TWindow(w,h,false,"CuMeshing");
  camera = new Camera1(w,h, M_PI/2.5);

  window->registerIoListener(camera);
  window->registerIoListener(this);

  do_init();
}

bool VizApp::renderFromThisThread() {
  bool out = false;

  window->startFrame();

  camera->update(.033f);
  if (wantToQuit) out = true;
  render();

  window->endFrame();
  return out;
}

void VizApp::internalLoop() {
  init();

  while (not doStop) {
    doStop = renderFromThisThread();
  }

  release();
}
void VizApp::joinThread() { internalThread.join(); }

void VizApp::render() {
  glClearColor(0,0,0,1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  camera->bind();

  do_render();
}

void VizApp::release() {
  do_release();
  delete camera;
  delete window;
}



void VizApp::reshapeFunc(int w, int h) {
}
void VizApp::keyboardFunc(int key, int scancode, int action, int mods) {
  keyDown[key-GLFW_KEY_A] = action != GLFW_RELEASE;
  if (key == GLFW_KEY_Q) wantToQuit = true;
}
void VizApp::clickFunc(int button, int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT and action == GLFW_PRESS) leftDown = true;
  if (button == GLFW_MOUSE_BUTTON_LEFT and action == GLFW_RELEASE) leftDown = false;
  if (button == GLFW_MOUSE_BUTTON_RIGHT and action == GLFW_PRESS) rightDown = true;
  if (button == GLFW_MOUSE_BUTTON_RIGHT and action == GLFW_RELEASE) rightDown = false;
}
void VizApp::motionFunc(double xpos, double ypos) {
  curX = xpos; curY = ypos;
}
