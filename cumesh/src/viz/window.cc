#include <Eigen/LU>

#include "window.h"

TWindow* TWindow::singleton = nullptr;
int TWindow::windowCnt = 0;
std::mutex TWindow::static_mtx;

static void _reshapeFunc(GLFWwindow* window, int w, int h) {
  TWindow::get()->reshapeFunc(window, w,h);
}
static void _keyboardFunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
  TWindow::get()->keyboardFunc(window, key, scancode, action, mods);
}
static void _clickFunc(GLFWwindow* window, int button, int action, int mods) {
  TWindow::get()->clickFunc(window, button, action, mods);
}
static void _motionFunc(GLFWwindow* window, double xpos, double ypos) {
  TWindow::get()->motionFunc(window, xpos, ypos);
}

TWindow::~TWindow() {
  glfwDestroyWindow(window);
  TWindow::singleton = nullptr;
}

TWindow::TWindow(int width, int height, bool headless, const std::string &title, bool egl)
  : width(width), height(height), headless(headless)
{
  std::lock_guard<std::mutex> lck(static_mtx);

  singleton = this;

  if (windowCnt == 0)
    if (!glfwInit()) {
      std::cerr << "Failed to initialize GLFW." << std::endl;
      glfwTerminate();
    }
  windowCnt++;

  std::string title_ = title + std::to_string(windowCnt);

  // NOTE: This doesn't affect non-default FBO!!!
  if (multisample) glfwWindowHint(GLFW_SAMPLES, 4);

  if (doubleBuffered)
    glfwWindowHint( GLFW_DOUBLEBUFFER, GL_TRUE );
  else
    glfwWindowHint( GLFW_DOUBLEBUFFER, GL_FALSE );

  if (egl)
    glfwWindowHint( GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);

  if (headless)
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

  window = glfwCreateWindow(width, height, title_.c_str(), NULL, NULL);
  if (window == NULL) {
    std::cerr << "Failed to open GLFW window: " << title_ << std::endl;
    glfwTerminate();
  }


  glfwMakeContextCurrent(window);
  glfwGetWindowSize(window, &width, &height);

  // Callback
  glfwSetWindowSizeCallback(window, &_reshapeFunc);
  glfwSetKeyCallback(window, &_keyboardFunc);
  glfwSetMouseButtonCallback(window, &_clickFunc);
  glfwSetCursorPosCallback(window, &_motionFunc);

  glewExperimental = true;  // This may be only true for linux environment.
  if (glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW." << std::endl;
    glfwTerminate();
  }

  if (multisample) glEnable(GL_MULTISAMPLE);
  else glDisable(GL_MULTISAMPLE);

  reshapeFunc(window, width, height);

}

void TWindow::destroy() {
  //for (auto it : ioListeners) if (it) delete it;
  glfwDestroyWindow(window);
  window = nullptr;
  TWindow::singleton = nullptr;
};

// NOTE:
// the mtx lock/unlock is *just* invalid.
// Although windowCnt is only touched in the constructor (behind the mtx),
// it could change between the if statement check and body...
void TWindow::startFrame() {
    if (windowCnt>1) static_mtx.lock();
    glfwMakeContextCurrent(window);
}
void TWindow::makeContextCurrent() {
  glfwMakeContextCurrent(window);
}
void TWindow::endFrame() {
    glfwPollEvents();
    glFlush();
    if (doubleBuffered) glfwSwapBuffers(window);

    if (windowCnt>1) static_mtx.unlock();
}

void TWindow::renderTestTriangle() {
  glBegin(GL_TRIANGLES);
  glColor4f(1.0,0.f,0.f,1.0f);
  glVertex3f(0.5f,0.f,0.f);
  glColor4f(0.0,1.f,0.f,1.0f);
  glVertex3f(0.f,.5f,0.f);
  glColor4f(0.0,0.f,1.f,1.0f);
  glVertex3f(0.f,0.f,0.5f);
  glEnd();
}




void TWindow::reshapeFunc(GLFWwindow* window, int w, int h) {

  width = w;
  height = h;

  for (const auto& io : ioListeners)
    io->reshapeFunc(w,h);
}
void TWindow::keyboardFunc(GLFWwindow* window, int key, int scancode, int action, int mods) {

  if (action == GLFW_PRESS || action == GLFW_REPEAT) {
    // Close window
    if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE) {
      glfwSetWindowShouldClose(window, GL_TRUE);
    }
  }

  for (const auto& io : ioListeners)
    io->keyboardFunc(key, scancode, action, mods);
}
void TWindow::clickFunc(GLFWwindow* window, int button, int action, int mods) {
  double x, y;
  glfwGetCursorPos(window, &x, &y);

  for (const auto& io : ioListeners)
    io->clickFunc(button, action, mods);
}

void TWindow::motionFunc(GLFWwindow* window, double xpos, double ypos) {

  for (const auto& io : ioListeners)
    io->motionFunc(xpos,ypos);
}


UsesIO::UsesIO() {
  // Disabled this, you now must manually call registerIoListener, which makes more sense
  /*
  if (TWindow::get() == nullptr) {
    std::cout << " WHAT: UsesIO() but no window!." << std::endl;
    exit(1);
  }
  TWindow::get()->registerIoListener(this);
  */
}
UsesIO::~UsesIO() {
  // Scene would break this if it were not for this check.
  // It's descturctor destroys TWindow (it has last ref to it), but UsesIO destructor then
  // tries to unregister itself!
  if (TWindow::get() != nullptr)
    TWindow::get()->unregisterIoListener(this);
}
