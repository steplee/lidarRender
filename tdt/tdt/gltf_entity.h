#pragma once

#include <unordered_map>
#include "tdt/gltf.h"

#define CheckGLErrors(desc)                                                                    \
    {                                                                                          \
        GLenum e = glGetError();                                                               \
        while (e != GL_NO_ERROR) {                                                                \
            printf("OpenGL error in \"%s\": %d (%d) %s:%d\n", desc, e, e, __FILE__, __LINE__); \
            fflush(stdout);                                                                    \
            exit(20);                                                                          \
        }                                                                                      \
    }


void matmul44(double C[16], const double A[16], const double B[16]);
void matmul44_colMajor(double C[16], const double A[16], const double B[16]);

class RenderState;
class RenderContext;

struct GltfEntity {
  GltfEntity();
  ~GltfEntity();

  void renderAllNodes(const RenderState& rs);
  void renderScene(const RenderState& rs, int i);
  void renderNode(const GltfNode& node, const RenderState& rs);

  void upload(const GltfModel& model);
  void destroy();

  uint32_t *vbos=0;
  uint32_t *texs=0;
  int nvbos=0, ntexs=0;

  std::vector<GltfNode> nodes;
  std::vector<GltfScene> scenes;
  std::vector<GltfMesh> meshes;
  std::vector<GltfPrimitive> prims;
  std::vector<GltfAccessor> accessors;
  std::vector<GltfBufferView> bufferViews;
  std::vector<GltfMaterial> materials;
};
