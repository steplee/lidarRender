#include "extra_entities.h"
#include <cassert>
#include "tdt/render_context.h"
#include <GL/glew.h>
#include <math.h>
#include <iostream>
#include "math.h"

void SphereEntity::init() {
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ibo);
  CheckGLErrors("post sphere gen buffer");

  int H = 24;
  int W = 32;

  float verts[H*W*(3)];
  double r = 1;
  int jj = 0;
  for (int y=0; y<H; y++)
  for (int x=0; x<W; x++) {
    double v = 1.0 * M_PI * ((double)y) / (H-1);
    double u = 2.0 * M_PI * ((double)x) / (W-1);
    verts[jj++] = r * sin(v) * sin(u);
    verts[jj++] = r * sin(v) * cos(u);
    verts[jj++] = r * cos(v);
  }

  ninds = (H-1) * (W-1) * 2 * 3;
  uint16_t inds[ninds];
  int ii=0;
  for (int y=0; y<H-1; y++)
  for (int x=0; x<W-1; x++) {
    // abc
    inds[ii++] = (y+0)*W + (x+0);
    inds[ii++] = (y+0)*W + (x+1);
    inds[ii++] = (y+1)*W + (x+1);
    // cda
    inds[ii++] = (y+1)*W + (x+1);
    inds[ii++] = (y+1)*W + (x+0);
    inds[ii++] = (y+0)*W + (x+0);
  }
  assert(ii == ninds);

  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float)*H*W*3, verts, GL_STATIC_DRAW);
  CheckGLErrors("post sphere buffer data 1");
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint16_t)*ninds, inds, GL_STATIC_DRAW);
  CheckGLErrors("post sphere buffer data 2");
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
void SphereEntity::destroy() {
  glDeleteBuffers(1, &vbo);
  glDeleteBuffers(1, &ibo);
}
void SphereEntity::render(RenderState& rs) {
  Shader* shader = &rs.ctx->basicUniformColorShader;
  glUseProgram(shader->id);
  CheckGLErrors("post shere use prog");

  double model[16];
  for (int i=0; i<16; i++) model[i] = i % 4 == i / 4;
  model[0] = model[5] = model[10] = radius;
  model[3] = pos[0]; model[7] = pos[1]; model[11] = pos[2];

  float mvp[16];
  matmul44_double_to_float(mvp, rs.proj, rs.modelView);

  glUniformMatrix4fv(shader->u_mvp, 1, true, mvp);
  CheckGLErrors("post sphere mvp bind");

  glUniform4fv(shader->u_color, 1, color);
  CheckGLErrors("post sphere color bind");

  glEnableVertexAttribArray(shader->in_pos);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(shader->in_pos, 3, GL_FLOAT, false, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);


  glDrawElements(GL_TRIANGLES, ninds, GL_UNSIGNED_SHORT, 0);
  CheckGLErrors("post sphere render");

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void SphereEntity::setPositionAndRadius(double pos_[3], double r_) {
  this->pos[0] = pos_[0];
  this->pos[1] = pos_[1];
  this->pos[2] = pos_[2];
  radius = r_;
}



void BoxEntity::init() {
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ibo);
  CheckGLErrors("post box gen buffer");

  int nverts = 8;
  double r = 1;
  float verts[nverts*3] = {
    0,0,0, // 0
    r,0,0, // 1
    r,r,0, // 2
    0,r,0, // 3
    0,0,r, // 4
    r,0,r, // 5
    r,r,r, // 6
    0,r,r };

  ninds = 12*3;
  uint16_t inds[ninds] = {
    1,0,2, 3,2,0,
    4,5,6, 6,7,4,

    // abcd = 0154
    0,1,5, 5,4,0,
    // abcd = 1265
    1,2,6, 6,5,1,
    // abcd = 3256
    2,3,6, 6,3,7,
    // abcd = 3047
    3,0,4, 4,7,3
  };

  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float)*nverts*3, verts, GL_STATIC_DRAW);
  CheckGLErrors("post box data 1");
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint16_t)*ninds, inds, GL_STATIC_DRAW);
  CheckGLErrors("post box data 2");
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
void BoxEntity::destroy() {
  glDeleteBuffers(1, &vbo);
  glDeleteBuffers(1, &ibo);
}
void BoxEntity::render(RenderState& rs) {
  Shader* shader = &rs.ctx->basicUniformColorShader;
  glUseProgram(shader->id);
  CheckGLErrors("post box use prog");

  float mvp[16];
  matmul44_double_to_float(mvp, rs.proj, rs.modelView);

  glUniformMatrix4fv(shader->u_mvp, 1, true, mvp);
  CheckGLErrors("post box mvp bind");

  glUniform4fv(shader->u_color, 1, color);
  CheckGLErrors("post box color bind");

  glEnableVertexAttribArray(shader->in_pos);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(shader->in_pos, 3, GL_FLOAT, false, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);


  glDrawElements(GL_TRIANGLES, ninds, GL_UNSIGNED_SHORT, 0);
  CheckGLErrors("post box render");

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
