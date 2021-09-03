#pragma once
#include "camera.h"
#include "window.h"

class ControllabeCamera :
  public Camera,
  public UsesIO
{
  public:

    ControllabeCamera(const CamSpec& cs);

    virtual void step(double dt);

    virtual void setPos(const double t[3]);
    virtual void setRot(const double R[9]);
    virtual void getPos(double t[3]);
    virtual void getRot(double R[9]);

  private:

    alignas(8) double q[4] = {1,0,0,0};
    alignas(8) double t[3] = {0,0,0};
    alignas(8) double avel[3] = {0,0,0};
    alignas(8) double vel[3] = {0,0,0};

    bool keyDown[26] = { false };
    bool shiftDown = false;
    bool ctrlDown = false;
    bool escDown = false;
    bool spaceDown = false;
    bool leftMouseDown = false, rightMouseDown = false;
    double lastX=-1, lastY=-1;
    double curX, curY;

    virtual void reshapeFunc(int w, int h) {};
    virtual void keyboardFunc(int key, int scancode, int action, int mods);
    virtual void clickFunc(int button, int action, int mods);
    virtual void motionFunc(double xpos, double ypos);
};

