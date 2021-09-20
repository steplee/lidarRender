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
  int u_view=-1;
  int u_modelView=-1;
  int u_invModelViewT=-1;
  int u_invMvp=-1;
  int u_sunPos=-1;

  uint32_t id=0;

  std::string name;

  inline Shader(const std::string& name) : name(name) {};

  void compile(uint32_t vertexShader, uint32_t fragShader);
  void compile(const std::string& vs, const std::string& fs);

  void findBindings();

};

struct RenderContext {
  Shader defaultGltfShader = Shader("defaultGltf");
  Shader basicUniformColorShader = Shader("basicUniformColor");
  Shader basicWhiteShader = Shader("basicWhite");
  Shader basicColorShader = Shader("basicColor");
  Shader basicTexturedShader = Shader("basicTextured");

  Shader litTextured = Shader("litTextured");
  Shader litUniformColored = Shader("litUniformColored");

  std::unordered_map<std::string, Shader> otherShaders;

  void compileShaders();
};

class SphereEntity;
class BoxEntity;
class Camera;

struct RenderState {
  RenderState(RenderContext* ctx);
  RenderState(const RenderState& rs);

  // These are ROW major
  alignas(8) float viewf[16];
  alignas(8) double modelView[16];
  alignas(8) double proj[16];
  int16_t w=0, h=0;
  Camera* cam = nullptr;

  SphereEntity* sphereEntity = nullptr;
  BoxEntity* boxEntity = nullptr;

  RenderContext* ctx;
};

// Class that helps with a render-to-texture + screen-space-effects pipeline
struct RenderEngine {
  uint32_t fbo;
  uint32_t frameRgbTex;
  uint32_t frameDepthTex;
  int w,h;

  void make(int w, int h);
  void setTarget();
  void unsetTarget();
  void renderToScreen(RenderContext& rs);
};

