#pragma once
#include <iostream>

#include <GL/glew.h>
#include <unordered_set>

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include <memory>
#include <mutex>

struct UsesIO;

class TWindow {

  public:
    TWindow(int width, int height, bool headless, const std::string& title="Viz", bool egl=false);
    ~TWindow();

    inline static TWindow* get() { return singleton; }
    inline void makeSingleton() { singleton = this; }
    void makeContextCurrent();

    void run();


    void reshapeFunc(GLFWwindow* window, int w, int h);
    void keyboardFunc(GLFWwindow* window, int key, int scancode, int action, int mods);
    void clickFunc(GLFWwindow* window, int button, int action, int mods);
    void motionFunc(GLFWwindow* window, double xpos, double ypos);

    inline void registerIoListener(UsesIO* io) { ioListeners.insert(io); };
    inline void unregisterIoListener(UsesIO* io) { if (ioListeners.find(io) != ioListeners.end()) ioListeners.erase(io); };

    inline bool shouldClose() { return glfwWindowShouldClose(window); }
    inline void setShouldClose() { glfwSetWindowShouldClose(window, GL_TRUE); }
    void renderTestTriangle();

    void startFrame();
    void endFrame();

    GLFWwindow *window;

    int width, height;

    void destroy();

  private:

    std::unordered_set<UsesIO*> ioListeners;

    bool headless;

    static TWindow* singleton;

    bool doubleBuffered = true;
    // NOTE: This doesn't affect non-default FBO!!!
    bool multisample = false;

    static int windowCnt;
    static std::mutex static_mtx;

};


/*
 * This class gets the singleton window and register itself to it.
 * So any class that extends this will listen to events.
 */

class UsesIO {
  public:
    UsesIO();
    UsesIO(const UsesIO&) =delete; // No copying, otherwise would invalidate register set
    ~UsesIO();

    virtual void reshapeFunc(int w, int h) {}
    virtual void keyboardFunc(int key, int scancode, int action, int mods) {}
    virtual void clickFunc(int button, int action, int mods) {}
    virtual void motionFunc(double xpos, double ypos) {}
};

