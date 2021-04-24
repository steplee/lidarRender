#pragma once

#include "common.h"
//#include "mesh.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;
//typedef CGAL::Triangulation_vertex_base_with_info_3<CGAL::Color, K> Vb;
typedef CGAL::Triangulation_vertex_base_with_info_3<int32_t, K> Vb;
//typedef CGAL::Triangulation_vertex_base_3<K> Vb;
typedef CGAL::Delaunay_triangulation_cell_base_3<K>                 Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb>                Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds>                      Delaunay;
typedef Delaunay::Point                                             Point;

struct DTOpts {
  bool createMesh = false;
};

struct DelaunayTetrahedrialization {
  inline DelaunayTetrahedrialization(const DTOpts& opts) : opts(opts) {}

  DTOpts opts;

  void run(RowMatrixCRef pts);
  RowMatrixX pts;

  Delaunay T;
  //IndexedMesh mesh;
};

