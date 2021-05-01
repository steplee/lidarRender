#include "las/read_las.h"
#include "recon/dt.h"
#include "recon/torch_stuff.h"
#include "geodset.h"

#include "recon/sorted_octree.h"

#include <GL/glew.h>
#include "viz/window.h"
#include "viz/camera.h"

#include <opencv2/highgui.hpp>


int main() {

  //std::string dir = "/data/pointCloudStuff/pc/laplata";
  std::string dir = "/data/pointCloudStuff/pc/dc";

  MultiLasDataset mld({dir});
  mld.split();

  //auto tile0 = mld.getTile(269, 3492);
  auto tile0 = mld.getTile(263,3529);

  //tile0.load(64);
  //tile0.load(1);
  tile0.load(2);
  //tile0.load(4);
  std::cout << " - Have " << tile0.pts.size() << " pts.\n";

  /*
  std::sort(tile0.pts.begin(), tile0.pts.end(), [](
        const Eigen::Vector3f& a,
        const Eigen::Vector3f& b) { return a(0) < b(0); });
  */
  //for (int i=0; i<tile0.pts.size(); i++) tile0.pts[i] *= .001f;

  /*
  std::stable_sort(tile0.pts.begin(), tile0.pts.end(), [](
        const Eigen::Vector3f& a,
        const Eigen::Vector3f& b) {
        return static_cast<int>(a(0)/50.f) < static_cast<int>(b(0)/50.f);
      });
      */

  /*
  SimpleGeoDataset dset("/data/pointCloudStuff/img/laplata/laplata.utm3.tif");
  cv::Mat img;
  dset.tlbrNative(tile0.tlbr, 1024,1024, img);
  cv::imshow("raster",img);
  cv::waitKey(0);
  */

  DelaunayTetrahedrialization dt({true});

  Eigen::Map<RowMatrixX> pts((float*)tile0.pts.data(), tile0.pts.size(), 3);
  //dt.run(pts);



#if 0
  std::cout << " - Computing normals." << std::endl;
  std::vector<Eigen::Vector3f> normals;
  std::vector<Eigen::Vector3f> basePts;
  estimateNormals(normals, basePts, tile0.pts);
  std::cout << " - COMPUTED " << normals.size() << " " << basePts.size() << "\n";

  std::vector<Eigen::Vector3f> ptNormalLines;
  ptNormalLines.resize(normals.size()*2);
  for (int i=0; i<normals.size(); i++) {
    ptNormalLines[i*2+0] = basePts[i];
    if (std::abs(normals[i](2)) < .4)
      ptNormalLines[i*2+1] = normals[i]*.01 + basePts[i];
    else
      ptNormalLines[i*2+1] = normals[i]*.01 + basePts[i];
  }

  TWindow window(1000,1000, false);
  Camera1 camera(1000,1000,M_PI/3.);
  glColor4f(1,1,1,1.);
  //dt.mesh.bake(true);
  for (int i=0; i<99999; i++) {
    window.startFrame();
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    camera.bind();
    camera.update(.03);

    //window.renderTestTriangle();
    //dt.mesh.render();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_DST_ALPHA);

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, basePts.data());
    //glVertexPointer(3, GL_FLOAT, 0, tile0.pts.data());
    glColor4f(0,1,0,.5);
    //glDrawArrays(GL_POINTS, 0, tile0.pts.size());
    glDrawArrays(GL_POINTS, 0, basePts.size());
    glVertexPointer(3, GL_FLOAT, 0, ptNormalLines.data());
    glColor4f(1,1,1,.18);
    glDrawArrays(GL_LINES, 0, ptNormalLines.size());
    glDisableClientState(GL_VERTEX_ARRAY);

    camera.unbind();
    window.endFrame();
  }
#else
  SortedOctree oct;
  std::vector<uint8_t> vis;
  std::vector<Eigen::Vector3f> basePts;
  std::vector<Eigen::Vector3f> basePts0, basePts1, ptsLines;
  //estimateVisibility(oct, vis, basePts, tile0.pts, 2);
  estimateVisibility(oct, vis, basePts, tile0.pts, 0);

  std::cout << " - Completing octree." << std::endl;
  completeOctree(oct);

  CompressedOctree coct;
  std::cout << " - Compressing octree." << std::endl;
  compressOctree(coct, oct);

  std::vector<float> rawNodes;
  //for (int i=2; i<oct.residentLvls; i++) getPointsAtNodeCenters(rawNodes, oct, i);
  getPointsAtCompressedNodeCenters(rawNodes, coct);

  //std::cout << " - RawNodes:\n";
  //for (int i=0; i<10; i++) std::cout << "   - " << rawNodes[i*3+0] << " " << rawNodes[i*3+1] << " " << rawNodes[i*3+2] << "\n";
  //for (int i=rawNodes.size()-30; i<10; i++) std::cout << "   - " << rawNodes[i*3+0] << " " << rawNodes[i*3+1] << " " << rawNodes[i*3+2] << "\n";

  //float ss = 3;
  float ss = 5;
  std::vector<Eigen::Vector3f> AngleLine = {
    Eigen::Vector3f{0,0,1},
    Eigen::Vector3f{-1,-1,ss}.normalized(),
    Eigen::Vector3f{ 1,-1,ss}.normalized(),
    Eigen::Vector3f{ 1, 1,ss}.normalized(),
    Eigen::Vector3f{-1, 1,ss}.normalized() };

  for (int i=0; i<basePts.size(); i++) {
    if (vis[i] == 0xff)
      basePts0.push_back(basePts[i]);
    else {
      basePts1.push_back(basePts[i]);
      /*
      for (int angle=0; angle<5; angle++) {
        if ((vis[i] & (1<<angle)) == 0) {
          ptsLines.push_back(basePts[i]);
          ptsLines.push_back(basePts[i] + AngleLine[angle] * .03);
        }
      }
      */

    }
  }



  TWindow window(1000,1000, false);
  Camera1 camera(1000,1000,M_PI/3.);
  glColor4f(1,1,1,1.);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA,GL_DST_ALPHA);

  GLuint vbos[5];
  glGenBuffers(3, vbos);
  glBindBuffer(GL_ARRAY_BUFFER, vbos[0]);
  glBufferData(GL_ARRAY_BUFFER, basePts0.size()*3*4, basePts0.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, vbos[1]);
  glBufferData(GL_ARRAY_BUFFER, basePts1.size()*3*4, basePts1.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, vbos[2]);
  glBufferData(GL_ARRAY_BUFFER, rawNodes.size()*4, rawNodes.data(), GL_STATIC_DRAW);

  for (int i=0; i<99999; i++) {
    window.startFrame();
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor4f(0,1,0,.5);
    //glVertexPointer(3, GL_FLOAT, 0, basePts0.data());
    glBindBuffer(GL_ARRAY_BUFFER, vbos[0]);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glDrawArrays(GL_POINTS, 0, basePts0.size());
    glColor4f(0,0,1,.5);
    //glVertexPointer(3, GL_FLOAT, 0, basePts1.data());
    glBindBuffer(GL_ARRAY_BUFFER, vbos[1]);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glDrawArrays(GL_POINTS, 0, basePts1.size());

    glColor4f(1,0,0,.5);
    //glVertexPointer(3, GL_FLOAT, 0, rawNodes.data());
    //glDrawArrays(GL_POINTS, 0, rawNodes.size()/3);
    glBindBuffer(GL_ARRAY_BUFFER, vbos[2]);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glDrawArrays(GL_POINTS, 0, rawNodes.size()/3);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    /*
    glVertexPointer(3, GL_FLOAT, 0, ptsLines.data());
    glColor4f(1,0,0,.02);
    glDrawArrays(GL_LINES, 0, ptsLines.size());
    */
    glDisableClientState(GL_VERTEX_ARRAY);

    camera.bind();
    camera.update(.03);

    camera.unbind();
    window.endFrame();
  }
#endif


  return 0;
}
