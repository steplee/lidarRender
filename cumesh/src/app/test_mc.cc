#include <iostream>

#include "heightmap/octree.h"
#include "heightmap/marching_cubes.h"

#include "viz/viz_app.h"
#include "viz/indexed_mesh.h"


struct MainViz : public VizApp {

  IndexedMesh mesh;
  IndexedMesh meshWireframe;
  IndexedMesh origPtsMesh;

  inline MainViz(const std::vector<float>& pts_, IndexedMesh& origPtsMesh_) : VizApp(1400,1000) {
    mesh.verts.resize(pts_.size()/3,3);
    for (int i=0; i<pts_.size(); i+=3) mesh.verts.row(i/3) << pts_[i+0], pts_[i+1], pts_[i+2];
    //mesh.mode = GL_POINTS;
    mesh.mode = GL_TRIANGLES;
    meshWireframe = convertTriangleMeshToLines(mesh);
    origPtsMesh = origPtsMesh_;
  }

  inline void do_init() override {
    mesh.bake(true);
    meshWireframe.bake(true);
    origPtsMesh.bake(true);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
  }
  inline void do_release() override {
    mesh.release();
    meshWireframe.release();
    origPtsMesh.release();
  }

  inline void do_render() override {
    glBegin(GL_LINES);
    glColor4f(1,0,0,1); glVertex3f(0,0,0); glVertex3f(1,0,0);
    glColor4f(0,1,0,1); glVertex3f(0,0,0); glVertex3f(0,1,0);
    glColor4f(0,0,1,1); glVertex3f(0,0,0); glVertex3f(0,0,1);
    glEnd();

    glColor4f(0,1,1,.5);
    /*
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, pts.data());
    glDrawArrays(GL_POINTS, 0, pts.size());
    glDisableClientState(GL_VERTEX_ARRAY);
    */
    mesh.render();
    glColor4f(.5,.5,.8,.2);
    meshWireframe.render();

    origPtsMesh.render();
  }
};

int main() {


  std::vector<float> inPts, inVals;
  int lvl = 7;
  float s  = 1 << lvl;
  if (0) {
    int N = 100000;
      for (int i=0; i<N; i++) {
        float n = 9999999;
        float xx,yy,zz;
        do {
          xx = ((rand() % 100000) / (100000.) + .0) * .5 * s;
          yy = ((rand() % 100000) / (100000.) + .0) * .5 * s;
          zz = ((rand() % 100000) / (100000.) + .0) * .5 * s;
          n = std::sqrt(xx*xx + yy*yy + zz*zz);
        } while (n < .3 * s);
        //} while (false);
        inPts.push_back(xx);
        inPts.push_back(yy);
        inPts.push_back(zz);
        inVals.push_back(1);
      }
  } else if (1) {
      int N_ = 200;
      for (int h=0; h<17; h++)
      for (int i=0; i<N_; i++)
      for (int j=0; j<N_; j++) {
        float r = .3 * (1. - ((float)h) * .015);
        float u = 2. * M_PI * ((float)i) / (N_-1);
        float v = 2. * M_PI * ((float)j) / (N_-1);
        float xx = .3 + r * std::cos(u) * std::sin(v);
        float yy = .3 + r * std::cos(u) * std::cos(v);
        float zz = .3 + r * std::sin(u);
        inPts.push_back(xx); inPts.push_back(yy); inPts.push_back(zz);
        float val = (h > 5 and h < 10) ? 1 : 0;
        inVals.push_back(val);
      }
  } else {
      int N_ = 64;
      lvl = 6; s = 1 << lvl;
      for (int h=0; h<N_; h++)
      for (int i=0; i<N_; i++)
      for (int j=0; j<N_; j++) {
        float w = (1./N_) + ((float)h) / (N_ + 3);
        float u = (1./N_) + ((float)i) / (N_ + 3);
        float v = (1./N_) + ((float)j) / (N_ + 3);
        float xx = u;
        float yy = v;
        float zz = w;

        inPts.push_back(xx); inPts.push_back(yy); inPts.push_back(zz);
        float val = 1;
        if (std::abs(w-.5) < .2) val = 0;
        if (std::abs(u-.5) < .2) val = 0;
        if (std::abs(v-.5) < .2) val = 0;
        //val = 1 - val;
        inVals.push_back(val);
      }
  }

  int N = inVals.size();
  SortedOctree<float> tree;
  tree.createOctree(inPts.data(), inVals.data(), N, 13, 2);

  std::vector<float> triPts;
  run_mc(tree, lvl, .5, triPts);
  //triPts = inPts;


  if (1) {
    // Show the octree decimated points instead of the initial ones
    inPts.resize(3*tree.lvlSizes[lvl]);
    inVals.resize(tree.lvlSizes[lvl]);
    std::vector<int32_t> inds(3*tree.lvlSizes[lvl]);
    tree.getCoordsValsHost(inds.data(), inVals.data(), lvl);
    for (int i=0; i<tree.lvlSizes[lvl]*3; i++) inPts[i] = inds[i] / s;
  }
  IndexedMesh origPtsMesh;
  origPtsMesh.verts.resize(inPts.size()/3,3);
  origPtsMesh.colors.resize(inPts.size()/3,4);
  origPtsMesh.mode = GL_POINTS;
  for (int i=0; i<inPts.size(); i+=3) origPtsMesh.verts.row(i/3) << inPts[i+0], inPts[i+1], inPts[i+2];
  for (int i=0; i<inVals.size(); i++) origPtsMesh.colors.row(i) << 1-inVals[i], inVals[i], 0, .5;

  MainViz viz(triPts, origPtsMesh);
  viz.startOnOwnThread();
  viz.joinThread();




  return 0;
}

