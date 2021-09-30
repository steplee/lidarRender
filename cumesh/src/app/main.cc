#include <iostream>
#include "las/read_las.h"
#include "viz/viz_app.h"
#include "viz/indexed_mesh.h"
#include "geo/geodset.h"
#include <cassert>

#include "heightmap/make.h"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>


struct MainViz : public VizApp {
  std::vector<LasPoint> pts;
  IndexedMesh mesh;
  IndexedMesh meshPts;
  cv::Mat waitingImg;

  /*
  inline MainViz(const std::vector<LasPoint>& pts_) : VizApp(900,900) {
    mesh.verts.resize(pts_.size(),3);
    for (int i=0; i<pts_.size(); i++) mesh.verts.row(i) << pts_[i].x, pts_[i].y, pts_[i].z;
    mesh.mode = GL_POINTS;
  }
  */
  inline MainViz(const HeightMapRasterizer& hmr, cv::Mat img, LasTile& tile) : VizApp(900,900) {
    waitingImg = img;
    //mesh.verts.resize(pts_.size(),3);
    //for (int i=0; i<pts_.size(); i++) mesh.verts.row(i) << pts_[i].x, pts_[i].y, pts_[i].z;
    
    if (0) {
    meshPts.verts.resize(hmr.inlyingPoints.size()/3,3);
    for (int i=0; i<hmr.inlyingPoints.size()/3; i++)
      meshPts.verts.row(i) <<  hmr.inlyingPoints[i*3+0], hmr.inlyingPoints[i*3+1], hmr.inlyingPoints[i*3+2];
    meshPts.mode = GL_POINTS;
    }

    mesh.verts.resize(hmr.meshVerts.size() / 3, 3);
    mesh.inds.resize(hmr.meshTris.size());
    for (int i=0; i<hmr.meshVerts.size()/3; i++)
      mesh.verts.row(i) << hmr.meshVerts[i*3+0], hmr.meshVerts[i*3+1], hmr.meshVerts[i*3+2];
    for (int i=0; i<hmr.meshTris.size(); i++)
      mesh.inds[i] = hmr.meshTris[i];

    if (waitingImg.empty()) {
      float max_z = mesh.verts.col(2).maxCoeff();
      mesh.colors.resize(mesh.verts.rows(), mesh.verts.cols());
      for (int i=0; i<hmr.meshVerts.size()/3; i++) {
        float z = mesh.verts(i,2)/max_z;
        mesh.colors.row(i) << 1, z, z*z;
      }
    } else {
      mesh.uvs.resize(hmr.meshVerts.size() / 3, 2);
      //Eigen::Vector2f min_ = mesh.verts.leftCols(2).colwise().minCoeff();
      //Eigen::Vector2f max_ = mesh.verts.leftCols(2).colwise().maxCoeff();
      //std::cout << " - Mesh min max " << min_.transpose() << " " << max_.transpose() << "\n";
      //Eigen::Vector2f min_ = tile.tlbr.head<2>().cast<float>();
      //Eigen::Vector2f max_ = tile.tlbr.tail<2>().cast<float>();
      for (int i=0; i<hmr.meshVerts.size()/3; i++)
        //mesh.uvs.row(i) << (hmr.meshVerts[i*3+0] - min_(0)) / (max_(0)-min_(0)),
                           //(hmr.meshVerts[i*3+1] - min_(1)) / (max_(1)-min_(1));
        mesh.uvs.row(i) << hmr.meshVerts[i*3+0], 1. - hmr.meshVerts[i*3+1];
    }

    mesh.mode = GL_TRIANGLES;
  }

  inline void do_init() override {

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    if (not waitingImg.empty()) {
      glGenTextures(1, &mesh.tex);
      glBindTexture(GL_TEXTURE_2D, mesh.tex);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, waitingImg.cols, waitingImg.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, waitingImg.data);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glBindTexture(GL_TEXTURE_2D, 0);
      waitingImg.release();
    }

    mesh.bake(true);
  }
  inline void do_release() override {
    mesh.release();
  }

  inline void do_render() override {
    glBegin(GL_LINES);
    glColor4f(1,0,0,1); glVertex3f(0,0,0); glVertex3f(1,0,0);
    glColor4f(0,1,0,1); glVertex3f(0,0,0); glVertex3f(0,1,0);
    glColor4f(0,0,1,1); glVertex3f(0,0,0); glVertex3f(0,0,1);
    glEnd();

    glColor4f(1,1,1,.8);
    /*
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, pts.data());
    glDrawArrays(GL_POINTS, 0, pts.size());
    glDisableClientState(GL_VERTEX_ARRAY);
    */

    mesh.render();
    meshPts.render();
  }
};

cv::Mat getRaster(SimpleGeoDataset& dset, const Eigen::Vector4d& tlbr) {
  Eigen::Vector4d bbox { tlbr(0), tlbr(1), tlbr(2) - tlbr(0), tlbr(3) - tlbr(1) };
  int w = 4096;
  //int h = (bbox(3) / bbox(2)) * w;
  int h = w;
  cv::Mat out;
  assert(dset.bboxNative(bbox, w,h,out));
  return out;
}

int main() {

  //MultiLasDataset ldset({"/data/pointCloudStuff/pc/laplata"});
  MultiLasDataset ldset({"/data/pointCloudStuff/pc/dc"});

  //SimpleGeoDataset rasterDset("/data/dc_tiffs/dc.tif");
  SimpleGeoDataset rasterDset("/data/dc_tiffs/dc0.tif");


  ldset.split();

  LasTile tile;

  {
    int y = 0, x = 1;
  //for (int y=0; y<ldset.heightInTiles(); y++) for (int x=0; x<ldset.widthInTiles(); x++) {
  //for (int y=0; y<1; y++) for (int x=0; x<1; x++) {
      tile = ldset.getRelativeTile(x,y);
      tile.load();
      std::cout << " tile " << x << " " << y << " has " << tile.pts.size() << " points.\n";
      std::cout << " tile " << x << " " << y << " has tlbr " << tile.tlbr.transpose() << "\n";
      for (int i=0; i<tile.pts.size(); i++)
        tile.pts[i] *= scaleFactor(tile.baseLvl);

      std::cout << " *** Running Heightmap Creator ***\n";
      HeightMapRasterizer hmr;
      hmr.run(tile.pts);

      //std::vector<LasPoint> newPts;
      //for (int i=0; i<hmr.inlyingPoints.size()/3; i++)
        //newPts.push_back({ hmr.inlyingPoints[i*3+0], hmr.inlyingPoints[i*3+1], hmr.inlyingPoints[i*3+2]});
      //MainViz viz(newPts);
      //MainViz viz(tile.pts);

      cv::Mat img = getRaster(rasterDset, tile.tlbr);

      MainViz viz(hmr, img, tile);

      viz.startOnOwnThread();
      viz.joinThread();
    }




  return 0;
}
