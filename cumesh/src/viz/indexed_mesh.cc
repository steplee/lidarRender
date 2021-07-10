#include "indexed_mesh.h"
#include <iostream>

#define BUFFER_OFFSET(x) ((void*)(x))

#define CheckGLErrors(desc)                                                                    \
    {                                                                                          \
        GLenum e = glGetError();                                                               \
        while (e != GL_NO_ERROR) {                                                                \
            printf("OpenGL error in \"%s\": %d (%d) %s:%d\n", desc, e, e, __FILE__, __LINE__); \
            fflush(stdout);                                                                    \
            exit(20);                                                                          \
        }                                                                                      \
    }

IndexedMesh::~IndexedMesh() {
  release();
}
void IndexedMesh::release() {
  if (vbo != 0) {
    glDeleteBuffers(1,&vbo);
    glDeleteBuffers(1,&ibo);
    vbo = 0; ibo = 0;
  }
  if (tex != 0) {
    glDeleteTextures(1,&tex);
    tex = 0;
  }
}

void IndexedMesh::bake(bool deallocateCpu) {
  upload(verts, vertexNormals, colors, uvs, inds);
  if (deallocateCpu) {
    vertexNormals.resize(0,0);
    uvs.resize(0,0);
    inds.clear();
    verts.resize(0,0);
    colors.resize(0,0);
  }
}

void IndexedMesh::upload(RowMatrixCRef pos, RowMatrixCRef normals, RowMatrixCRef colors, RowMatrixCRef uvs, const std::vector<uint32_t>& inds) {
  //glewInit();

  haveInds = inds.size() > 0;
  vertCnt = pos.rows();

  int cols = pos.cols() + normals.cols() + uvs.cols() + colors.cols();
  vboStride = cols * 4;
  RowMatrixXf verts(pos.rows(), cols);
  verts.block(0 , 0                                       , pos.rows() , pos.cols())     = pos;
  if (normals.cols() > 0)
    verts.block(0 , pos.cols()                              , pos.rows() , normals.cols()) = normals;
  if (colors.cols() > 0)
    verts.block(0 , normals.cols()+pos.cols()               , pos.rows() , colors.cols())  = colors;
  if (uvs.cols() > 0)
    verts.block(0 , colors.cols()+normals.cols()+pos.cols() , pos.rows() , uvs.cols())     = uvs;

  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ibo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
  glBufferData(GL_ARRAY_BUFFER, verts.rows()*verts.cols()*4, verts.data(), GL_STATIC_DRAW);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, inds.size()*4, inds.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  vboHasNormal = normals.rows() > 1;
  vboHasUv = uvs.rows() > 1;
  vboHasColor = colors.rows() > 1;
  iboIndCnt = inds.size();
}

void IndexedMesh::print() {
  std::cout << " - IndexedMesh:\n\tverts: " << verts.rows() << "\n\tinds: " << inds.size();
  std::cout << "\n\tvertexNormals: " << vertexNormals.rows();
  std::cout << "\n\tfaceNormals: " << faceNormals.rows();
  std::cout << "\n\tuvs: " << uvs.rows() << " x " << uvs.cols();
  std::cout << "\n\ttex: " << tex;
  std::cout << "\n";
}


void IndexedMesh::render() {
  if (inds.size() > 0) haveInds = true;
  if (verts.rows() > 0) vertCnt = verts.rows(); // messy, but works here.

  if (vbo != 0) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    glEnableClientState(GL_VERTEX_ARRAY);
    uint64_t offset = 0;
    glVertexPointer(3, GL_FLOAT, vboStride, BUFFER_OFFSET(offset));
    offset += 4*3;

    if (vboHasColor) {
      glEnableClientState(GL_COLOR_ARRAY);
      glColorPointer(4, GL_FLOAT, vboStride, BUFFER_OFFSET(offset));
      offset += 4 * 3;
    }
    if (vboHasNormal) {
      glEnableClientState(GL_NORMAL_ARRAY);
      glNormalPointer(GL_FLOAT, vboStride, BUFFER_OFFSET(offset));
      offset += 4*3;
    }
    if (vboHasUv and tex != 0) {
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, tex);
      glEnableClientState(GL_TEXTURE_COORD_ARRAY);
      glTexCoordPointer(2, GL_FLOAT, vboStride, BUFFER_OFFSET(offset));
      offset += 4*2;
    }

    if (haveInds)
      glDrawElements(mode, iboIndCnt, GL_UNSIGNED_INT, 0);
    else
      glDrawArrays(mode, 0, vertCnt);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  }


  else {
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, verts.data());

    if (vertexNormals.size()) {
      assert(vertexNormals.size() == verts.size());
      glEnableClientState(GL_NORMAL_ARRAY);
      glNormalPointer(GL_FLOAT, 0, vertexNormals.data());
    }

    if (colors.size()) {
      glEnableClientState(GL_COLOR_ARRAY);
      glColorPointer(4, GL_FLOAT, 0, colors.data());
      glColor4f(1,1,1,1.);
    }

    if (uvs.size() and tex != 0) {
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, tex);
      //glClientActiveTexture(GL_TEXTURE0);
      glEnableClientState(GL_TEXTURE_COORD_ARRAY);
      glTexCoordPointer(2, GL_FLOAT, 0, (void*)uvs.data());
    }

    if (haveInds)
      glDrawElements(mode, inds.size(), GL_UNSIGNED_INT, (void*)inds.data());
    else glDrawArrays(mode, 0, vertCnt);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
  }
}

IndexedMesh convertTriangleMeshToLines(const IndexedMesh& in) {
  IndexedMesh out;
  out.verts = in.verts;
  if (in.inds.size() == 0) {
    out.inds.reserve(in.inds.size()*2);
    for (int i=0; i<in.verts.size(); i+=3) {
      int a = i, b = i+1, c = i+2;
      out.inds.push_back(a); out.inds.push_back(b); out.inds.push_back(a);
      out.inds.push_back(c); out.inds.push_back(b); out.inds.push_back(c);
    }
  } else {
    out.inds.reserve(in.inds.size()*2);
    for (int i=0; i<in.inds.size(); i+=3) {
      int a = in.inds[i], b = in.inds[i+1], c = in.inds[i+2];
      out.inds.push_back(a); out.inds.push_back(b); out.inds.push_back(a);
      out.inds.push_back(c); out.inds.push_back(b); out.inds.push_back(c);
    }
  }
  out.mode = GL_LINES;
  return out;
}


IndexedMesh normalsToMesh(const RowMatrixCRef verts, const RowMatrixCRef normals, float size) {
  IndexedMesh out;
  out.verts.resize(verts.rows()*2, 3);
  for (int i=0; i<verts.rows(); i++) {
    Vector3 a = verts.row(i);
    Vector3 b = verts.row(i) + normals.row(i) * size;
    out.verts.row(i*2  ) = a;
    out.verts.row(i*2+1) = b;
    out.inds.push_back(2*i  );
    out.inds.push_back(2*i+1);
  }
  out.mode = GL_LINES;
  return out;
}

