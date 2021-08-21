#include "tdt/window.h"

#include "tdt/gltf.h"
#include "tdt/gltf_entity.h"
#include "tdt/render_context.h"
#include "tdt/tdtiles/parse.h"
#include "tdt/extra_entities.h"

#include <unistd.h>
#include <iostream>


int main(int argc, char** argv) {
  std::string fname { argv[1] };
  std::string dir = fname.substr(0, fname.rfind("/")+1);

  TWindow window(900,900, false, "Test1");

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  double scale = 200;
  double n = scale * .01, r = scale, f = scale * 4;
  //double n = 5, r = 120, f = 400; // for Fox
  //double n = .001, r = .05, f = 1; // for ToyCar
  //double n = .1, r = 2, f = 20; // for Suzanne
  //double n = 20, r = 520, f = 1200; // for 2CylinderEngine
  double uv = .5;
  glFrustumf(-n*uv,n*uv,-n*uv,n*uv, n, f);
  double proj[16];
  glGetDoublev(GL_PROJECTION_MATRIX, proj); // because im lazy
  CheckGLErrors("post get proj");

  //auto t = TileBase::fromFile(fname);
  //auto t = TileBase::fromFile(fname);
  TileRoot root(dir, fname);
  //delete t;

  RenderContext rctx;
  rctx.compileShaders();

  SphereEntity sphereEnt; sphereEnt.init();
  BoxEntity boxEnt; boxEnt.init();

  glEnable(GL_CULL_FACE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


  for (double t=0; t<8*3.141; t += .01) {
    glMatrixMode(GL_MODELVIEW);
    double x = std::sin(t) * r;
    double y = std::cos(t) * r;
    double z = r*.2 + std::cos(t * .2) * r * .2;
    glLoadIdentity();
    //std::swap(y,z);
    //gluLookAt(x,y,z, 0,0,0, 0,1,0);
    gluLookAt(x,y,z, 0,0,0, 0,0,1);
    double view[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, view);
    CheckGLErrors("post get view");

    window.startFrame();
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    RenderState rs(&rctx);
    rs.sphereEntity = &sphereEnt;
    rs.boxEntity = &boxEnt;
    matmul44(rs.mvp, view, proj);
    for (int i=0; i<4; i++) for (int j=0; j<i; j++) std::swap(rs.mvp[i*4+j], rs.mvp[j*4+i]);
    //entity.renderScene(rs, 0);
    root.render(rs);

    //sphereEnt.render(rs);
    //double p[3] = { 0, 0, t  * 2};
    //double r = t + 1;
    //sphereEnt.setPositionAndRadius(p,r);
    //boxEnt.render(rs);

    window.endFrame();
    //std::cout << " - render.\n";
  }

  return 0;
}

