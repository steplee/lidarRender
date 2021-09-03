#include "tdt/window.h"

#include "tdt/gltf.h"
#include "tdt/gltf_entity.h"
#include "tdt/render_context.h"
#include "tdt/tdtiles/parse.h"
#include "tdt/extra_entities.h"

#include "tdt/controllable_camera.h"
#include "tdt/tdtiles/sight.h"

#include <unistd.h>
#include <iostream>

int main(int argc, char** argv) {
  std::string fname { argv[1] };
  std::string dir = fname.substr(0, fname.rfind("/")+1);

  int w = 900, h = 900;
  TWindow window(w,h, false, "Test1");

  ControllabeCamera cam(CamSpec(45,45,1));

  RenderContext rctx;
  rctx.compileShaders();

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);

#if 0
  while (true) {

    cam.step(1.0 / 33.0);
    RenderState rs(&rctx);
    cam.viewProj(rs.mvp);
    //for (int i=0; i<16; i++) rs.mvp[i] = i % 5 == 0;

    window.startFrame();
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //glBegin(GL_LINES);
    //glColor4f(1,0,0,1); glVertex3f(0,0,0); glVertex3f(1,0,0);
    //glColor4f(0,1,0,1); glVertex3f(0,0,0); glVertex3f(0,1,0);
    //glColor4f(0,0,1,1); glVertex3f(0,0,0); glVertex3f(0,0,1);
    //glEnd();
    float mvp[16]; for (int i=0; i<16; i++) mvp[i] = rs.mvp[i];
    auto shader = &rs.ctx->basicUniformColorShader;
    glUseProgram(shader->id);
    static const float verts[] = {
      0,0,0, 1,0,0,
      0,0,0, 0,1,0,
      0,0,0, 0,0,1 };
    glEnableVertexAttribArray(shader->in_pos);
    glVertexAttribPointer(shader->in_pos, 3, GL_FLOAT, false, 0, verts);
    glUniformMatrix4fv(shader->u_mvp, 1, true, mvp);
    glUniform4f(shader->u_color, 1,0,0,1); glDrawArrays(GL_LINES, 0, 2);
    glUniform4f(shader->u_color, 0,1,0,1); glDrawArrays(GL_LINES, 2, 2);
    glUniform4f(shader->u_color, 0,0,1,1); glDrawArrays(GL_LINES, 4, 2);


    window.endFrame();
  }
#elif 1
  Sight sight;
  sight.addTileset(dir, fname);

  int ii = 0;
  while (true) {

    cam.step(1.0 / 33.0);
    RenderState rs(&rctx);
    cam.viewProj(rs.mvp);
    rs.w = w, rs.h = h;
    //const auto spec = cam.getSpec();
    //rs.u = spec.u, rs.v = spec.v;
    rs.cam = &cam;

    window.startFrame();
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (ii < 5 or ii % 5 == 0)
      sight.update(rs);
    sight.render(rs);

    window.endFrame();
    ii++;
  }
#endif

  /*
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
  */
};
