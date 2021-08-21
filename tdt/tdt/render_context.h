#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

#define CheckGLErrors(desc)                                                                    \
    {                                                                                          \
        GLenum e = glGetError();                                                               \
        while (e != GL_NO_ERROR) {                                                                \
            printf("OpenGL error in \"%s\": %d (%d) %s:%d\n", desc, e, e, __FILE__, __LINE__); \
            fflush(stdout);                                                                    \
            exit(20);                                                                          \
        }                                                                                      \
    }

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

  std::unordered_map<std::string, Shader> otherShaders;

  void compileShaders();
};

class SphereEntity;
class BoxEntity;

struct RenderState {
  RenderState(RenderContext* ctx);
  RenderState(const RenderState& rs);

  // These are ROW major
  double mvp[16];

  SphereEntity* sphereEntity = nullptr;
  BoxEntity* boxEntity = nullptr;

  RenderContext* ctx;
};

