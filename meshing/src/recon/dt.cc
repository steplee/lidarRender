#include "dt.h"



void DelaunayTetrahedrialization::run(RowMatrixCRef pts_) {
  this->pts = pts_;

  std::cout << " - Triangulating " << pts.rows() << " pts.\n";
  for (int i=0; i<pts.rows(); i++) {
    // CGAL does not store vertex id, but we want it to make it easier to form indexed mesh.
    auto vh = T.insert(Point(pts(i,0),pts(i,1),pts(i,2)));
    vh->info() = i;
    if (i % (pts.rows()/20) == 0) std::cout << "    - " << (((float)i)/pts.rows())*100. <<  "% (" << i << "/" << pts.rows() << ")\n";
  }

  std::cout << " - Got "
    << T.number_of_cells() << " cells "
    << T.number_of_facets() << " facets "
    << T.number_of_edges() << " edges "
    << T.number_of_vertices() << " verts.\n";

  if (not opts.createMesh) return;
  std::cout << " - DelaunayTetrahedrialization::run making mesh." << std::endl;

#if 1

  //for (auto it = T.finite_facets_begin(), end = T.finite_facets_end(); it != end; ++it) { }
  int ii = 0;
  #define PUSH_TRI(x,y,z) mesh.inds.push_back((x)); mesh.inds.push_back((y)); mesh.inds.push_back((z));
  for (auto it = T.finite_cells_begin(), end = T.finite_cells_end(); it != end; ++it) {
    int32_t a = it->vertex(0)->info();
    int32_t b = it->vertex(1)->info();
    int32_t c = it->vertex(2)->info();
    int32_t d = it->vertex(3)->info();
    // This is the proper wind order.
    PUSH_TRI(b,a,c);
    PUSH_TRI(d,c,a);
    PUSH_TRI(b,c,d);
    PUSH_TRI(d,a,b);
    //if (ii > 5000000) break;
    ii++;
  }

  /*
  RowMatrix vv(1000,3);
  ii = 0;
  for (auto it = T.finite_vertices_begin(), end = T.finite_vertices_end(); it != end; ++it) {
    auto pt = it->point();
    if (ii >= vv.rows()) vv.conservativeResize(vv.rows()*2, Eigen::NoChange);
    vv(ii, 0) = pt[0];
    vv(ii, 1) = pt[1];
    vv(ii, 2) = pt[2];
    ii++;
  }
  mesh.verts = vv.topRows(ii);
  */
  mesh.verts = pts;
  mesh.mode = GL_TRIANGLES;

  RowMatrixXf colors(pts.rows(), 4);
  for (int i=0; i<pts.rows(); i++) {
    Vector3 cc = Vector3::Random();
    colors.row(i).leftCols(3) = cc.normalized();
    colors(i,3) = .2;
  }
  mesh.colors = colors;
#endif
}

