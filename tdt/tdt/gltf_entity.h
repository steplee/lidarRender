#pragma once

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

struct Shader {
  int in_pos=-1;
  int in_normal=-1;
  int in_texcoord0=-1;
  int in_texcoord1=-1;
  int in_tangent=-1;
  int in_color=-1;
  int in_size=-1;
  int u_diffuseTex=-1;
  int u_color=-1;
  int u_mvp=-1;
  int u_proj=-1;
  int u_modelView=-1;
  int u_invModelView=-1;
  int u_invMvp=-1;
  int u_sunPos=-1;

  uint32_t id=0;

  std::string name;

  inline Shader(const std::string& name) : name(name) {};

  void compile(uint32_t vertexShader, uint32_t fragShader);
  void findBindings();

};

struct RenderContext {
  Shader defaultGltfShader = Shader("defaultGltf");
  Shader basicUniformColorShader = Shader("basicUniformColor");
  Shader basicWhiteShader = Shader("basicWhite");

  void compileShaders();
};

struct RenderState {
  RenderState(const RenderContext* ctx);
  RenderState(const RenderState& rs);

  // These are ROW major
  double mvp[16];

  const RenderContext* ctx;
};


struct GltfEntity {
  GltfEntity();
  ~GltfEntity();

  void render(const RenderState& rs);
  void renderNode(const GltfNode& node, const RenderState& rs);

  void upload(const GltfModel& model);
  void destroy();

  uint32_t *vbos=0;
  uint32_t *texs=0;
  int nvbos=0, ntexs=0;

  std::vector<GltfNode> nodes;
  std::vector<GltfMesh> meshes;
  std::vector<GltfPrimitive> prims;
  std::vector<GltfAccessor> accessors;
  std::vector<GltfBufferView> bufferViews;
  std::vector<GltfMaterial> materials;
};
