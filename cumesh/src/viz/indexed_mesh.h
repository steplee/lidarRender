#pragma once

#include "common.h"

struct IndexedMesh {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ~IndexedMesh();

  RowMatrixXf verts, vertexNormals, faceNormals, colors, uvs;
  std::vector<uint32_t> inds;
  bool haveInds = false;

  GLuint tex = 0;
  int vertCntr = 0;
  GLenum mode = GL_TRIANGLES;

  void render();
  void print();
  void release();

  // cpu -> gpu transfer.
  void upload(RowMatrixCRef verts, RowMatrixCRef normals, RowMatrixCRef colors, RowMatrixCRef uvs, const std::vector<uint32_t>& inds);
  void bake(bool deallocateCpu);
  GLuint vbo=0, ibo=0, iboIndCnt=0, vertCnt=0;
  GLsizei vboStride=0;
  bool vboHasUv = false, vboHasNormal = false, vboHasColor = false;
};

// Create a mesh to display the edges of a triangular mesh.
IndexedMesh convertTriangleMeshToLines(const IndexedMesh&);
// Create a mesh to display the normals of a point set.
IndexedMesh normalsToMesh(const RowMatrixCRef verts, const RowMatrixCRef normals, float size);


