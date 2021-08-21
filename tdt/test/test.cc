#include "tdt/window.h"

#include "tdt/gltf.h"
#include "tdt/gltf_entity.h"
#include "tdt/render_context.h"

#include <unistd.h>
#include <iostream>


int main(int argc, char** argv) {

  std::string fname { argv[1] };
  GltfModel* model = GltfModel::fromFile(fname);

  double scale = std::atof( argv[2] );

  TWindow window(900,900, false, "Test1");
  CheckGLErrors("post create window");

  //sleep(1);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  double n = scale * .001, r = scale, f = scale * 4;
  //double n = 5, r = 120, f = 400; // for Fox
  //double n = .001, r = .05, f = 1; // for ToyCar
  //double n = .1, r = 2, f = 20; // for Suzanne
  //double n = 20, r = 520, f = 1200; // for 2CylinderEngine
  glFrustumf(-n,n,-n,n, n, f);
  double proj[16];
  glGetDoublev(GL_PROJECTION_MATRIX, proj); // because im lazy
  CheckGLErrors("post get proj");

  RenderContext rctx;
  rctx.compileShaders();

  GltfEntity entity;
  entity.upload(*model);
  CheckGLErrors("post upload");

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);

  for (double t=0; t<8*3.141; t += .01) {
    glMatrixMode(GL_MODELVIEW);
    double x = std::sin(t) * r;
    double y = std::cos(t) * r;
    double z = r*.2 + std::cos(t * .2) * r * .2;
    std::swap(y,z);
    glLoadIdentity();
    gluLookAt(x,y,z, 0,0,0, 0,1,0);
    double view[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, view);
    CheckGLErrors("post get view");

    window.startFrame();
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBegin(GL_LINES);
    glColor4f(0,0,1,.5); glVertex3f(0,0,0); glVertex3f(0,0,20);
    glColor4f(1,0,0,.5); glVertex3f(0,0,0); glVertex3f(20,0,0);
    glColor4f(0,1,0,.5); glVertex3f(0,0,0); glVertex3f(0,20,0);
    glEnd();
    glColor4f(1,1,1,1);
    //glBegin(GL_TRIANGLES); glVertex3f(0,0,20); glVertex3f(20,0,0); glVertex3f(0,20,0); glEnd();

    RenderState rs(&rctx);
    // C = A*B
    // C = (At*Bt)t
    matmul44(rs.mvp, view, proj);
    for (int i=0; i<4; i++) for (int j=0; j<i; j++) std::swap(rs.mvp[i*4+j], rs.mvp[j*4+i]);
    entity.renderScene(rs, 0);


    window.endFrame();
    std::cout << " - render.\n";
  }



  return 0;
}
