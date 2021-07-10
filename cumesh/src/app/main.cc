#include <iostream>
#include "las/read_las.h"
#include "viz/viz_app.h"
#include "viz/indexed_mesh.h"

#include "heightmap/make.h"


struct MainViz : public VizApp {
  std::vector<LasPoint> pts;
  IndexedMesh mesh;
  inline MainViz(const std::vector<LasPoint>& pts_) : VizApp(900,900) {
    //pts = pts_;
    mesh.verts.resize(pts_.size(),3);
    //for (int i=0; i<pts_.size(); i++) mesh.verts.row(i) << pts_[i].transpose();
    for (int i=0; i<pts_.size(); i++) mesh.verts.row(i) << pts_[i].x, pts_[i].y, pts_[i].z;
    mesh.mode = GL_POINTS;
  }

  inline void do_init() override {
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

    glColor4f(0,1,1,.8);
    /*
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, pts.data());
    glDrawArrays(GL_POINTS, 0, pts.size());
    glDisableClientState(GL_VERTEX_ARRAY);
    */
    mesh.render();
  }
};

int main() {

  //MultiLasDataset ldset({"/data/pointCloudStuff/pc/laplata"});
  MultiLasDataset ldset({"/data/pointCloudStuff/pc/dc"});

  ldset.split();

  LasTile tile;

  {
    int y = 0, x = 3;
  //for (int y=0; y<ldset.heightInTiles(); y++) for (int x=0; x<ldset.widthInTiles(); x++) {
  //for (int y=0; y<1; y++) for (int x=0; x<1; x++) {
      tile = ldset.getRelativeTile(x,y);
      tile.load();
      std::cout << " tile " << x << " " << y << " has " << tile.pts.size() << " points.\n";
      for (int i=0; i<tile.pts.size(); i++)
        tile.pts[i] *= scaleFactor(tile.baseLvl);

      std::cout << " *** Running Heightmap Creator ***\n";
      HeightMapRasterizer hmr;
      hmr.run(tile.pts);

      std::vector<LasPoint> newPts;
      for (int i=0; i<hmr.inlyingPoints.size()/3; i++)
        newPts.push_back({ hmr.inlyingPoints[i*3+0], hmr.inlyingPoints[i*3+1], hmr.inlyingPoints[i*3+2]});
      MainViz viz(newPts);
      //MainViz viz(tile.pts);

      viz.startOnOwnThread();
      viz.joinThread();
    }




  return 0;
}
