#pragma once
#include "camera.h"
#include "window.h"
#include <thread>

class VizApp : public UsesIO {
  public:
    VizApp(int w, int h);



    // User must call renderFromThisThread(), and release() at end.
    void startOnThisThread();

    // Will manage render loop itself!
    void startOnOwnThread();

    bool renderFromThisThread(); // returns true if should stop.
    void release();

    void joinThread();

  protected:
    int w, h;
    bool haveThread = false;
    bool doStop = false;

    virtual void do_init() {};
    virtual void do_render() {};
    virtual void do_release() {};

    TWindow* window=nullptr;
    Camera1* camera=nullptr;

    double lastX=-1, lastY=-1, curX=-1, curY=-1;
    bool leftDown=false, rightDown=false;
    bool keyDown[26] = {false};
    bool wantToQuit = false;

    virtual void reshapeFunc(int w, int h) override;
    virtual void keyboardFunc(int key, int scancode, int action, int mods) override;
    virtual void clickFunc(int button, int action, int mods) override;
    virtual void motionFunc(double xpos, double ypos) override;

  private:
    void init();

    std::thread internalThread;
    void internalLoop();

    void render();

};

