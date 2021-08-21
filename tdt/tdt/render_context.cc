//#include "gltf_entity.h"
#include "tdt/render_context.h"
#include <GL/glew.h>
#include <iostream>
#include <cstring>


// -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 
// Shader Stuff
// -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 

bool LoadShaderFromString(const std::string& name, GLenum shaderType, GLuint &shader, const std::string& src) {
  const GLchar *srcs[1];
  srcs[0] = &src[0];
  GLint val = 0;

  shader = glCreateShader(shaderType);
  glShaderSource(shader, 1, srcs, NULL);
  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, &val);
  if (val != GL_TRUE) {
    char log[4096];
    GLsizei msglen;
    glGetShaderInfoLog(shader, 4096, &msglen, log);
    printf("Shader compilation failed for shader %s\n", name.c_str());
    printf("Shader log: %s\n", log);
    exit(1);
  }

  return true;
}

bool LoadShaderFromFile(const std::string& name, GLenum shaderType, GLuint &shader, const char *shaderSourceFilename) {
  GLint val = 0;
  if (shader != 0) glDeleteShader(shader);

  std::string srcbuf;
  FILE *fp = fopen(shaderSourceFilename, "rb");
  if (!fp) {
    printf("Failed to load shader %s\n", shaderSourceFilename);
    exit(1);
  }
  fseek(fp, 0, SEEK_END);
  size_t len = ftell(fp);
  rewind(fp);
  srcbuf.resize(len + 1);
  len = fread(&srcbuf[0], 1, len, fp);
  srcbuf[len] = 0;
  fclose(fp);

  return LoadShaderFromString(name, shaderType, shader, srcbuf);
}

bool LinkShader(const std::string &name, GLuint &prog, GLuint &vertShader, GLuint &fragShader) {
  GLint val = 0;

  if (prog != 0) {
    glDeleteProgram(prog);
  }

  prog = glCreateProgram();

  glAttachShader(prog, vertShader);
  glAttachShader(prog, fragShader);
  glLinkProgram(prog);

  glGetProgramiv(prog, GL_LINK_STATUS, &val);
  if (val != GL_TRUE) {
    std::cout << " - Error compiling shader " << name << ":\n";
    exit(1);
  }

  return true;
}


void Shader::compile(uint32_t vertexId, uint32_t fragId) {
    printf(" - Creating shader '%s'.\n", name.c_str());

    CheckGLErrors("pre link");
    LinkShader(name, id, vertexId, fragId);
    printf("Link shader '%s' OK.\n", name.c_str());
    CheckGLErrors("post link");

    glUseProgram(id);
    findBindings();
    glUseProgram(0);
    CheckGLErrors("post find bindings");
}
#define GET_UNIFORM(a) a = glGetUniformLocation(id, #a );
#define GET_ATTRIB(a) a = glGetAttribLocation(id, #a );
void Shader::findBindings() {
  CheckGLErrors("pre-find bindings.");

  GET_UNIFORM(u_mvp);
  GET_UNIFORM(u_modelView);
  GET_UNIFORM(u_invModelView);
  GET_UNIFORM(u_sunPos);
  GET_UNIFORM(u_diffuseTex);
  GET_UNIFORM(u_color);

  in_pos = glGetAttribLocation(id, "in_pos");
  std::cout << " - shader in_pos " << in_pos << "\n";
  //GET_ATTRIB(in_pos)
  GET_ATTRIB(in_texcoord0)
  GET_ATTRIB(in_texcoord1)
  GET_ATTRIB(in_tangent)
  GET_ATTRIB(in_color)
  GET_ATTRIB(in_size)
  GET_ATTRIB(in_normal)
  CheckGLErrors("post-find bindings.");
}

const std::string basicVertSrc = R"(
#version 440
in vec3 in_pos;
uniform mat4 u_mvp;
void main() {
  gl_Position = u_mvp * vec4(in_pos,1.0);
}
)";
const std::string basicFragSrc = R"(
#version 440
out vec4 outColor;
void main() {
  outColor = vec4(1.0, 1.0, 1.0, 0.4);
}
)";
const std::string basicUniformColorFragSrc = R"(
#version 440
out vec4 outColor;
uniform vec4 u_color;
void main() {
  outColor = u_color;
}
)";

const std::string basicTexturedVertSrc = R"(
#version 440
in vec3 in_pos;
in vec2 in_texcoord0;
out vec2 v_uv;
uniform mat4 u_mvp;
void main() {
  gl_Position = u_mvp * vec4(in_pos,1.0);
  v_uv = in_texcoord0;
}
)";
const std::string basicTexturedFragSrc = R"(
#version 440
uniform sampler2D u_diffuse;
in vec2 v_uv;
out vec4 outColor;
void main() {
  outColor = texture(u_diffuse, v_uv);
}
)";

void RenderContext::compileShaders() {


  CheckGLErrors("pre compile");
  GLuint basicVertId = 0;
  GLuint basicWhiteFragId = 0;
  GLuint basicUniformColorFragId = 0;
  LoadShaderFromString("basicVert", GL_VERTEX_SHADER, basicVertId, basicVertSrc);
  LoadShaderFromString("basicWhiteFrag", GL_FRAGMENT_SHADER, basicWhiteFragId, basicFragSrc);
  LoadShaderFromString("basicUniformColorFrag", GL_FRAGMENT_SHADER, basicUniformColorFragId, basicUniformColorFragSrc);
  GLuint basicTexturedVert = 0;
  GLuint basicTexturedFrag = 0;
  LoadShaderFromString("basicTexturedVert", GL_VERTEX_SHADER, basicTexturedVert, basicTexturedVertSrc);
  LoadShaderFromString("basicTexturedWhiteFrag", GL_FRAGMENT_SHADER, basicTexturedFrag, basicTexturedFragSrc);

  basicWhiteShader.compile(basicVertId, basicWhiteFragId);
  basicUniformColorShader.compile(basicVertId, basicUniformColorFragId);

  defaultGltfShader.compile(basicTexturedVert, basicTexturedFrag);

  glDeleteShader(basicVertId); glDeleteShader(basicWhiteFragId);
  glDeleteShader(basicUniformColorFragId);
  glDeleteShader(basicTexturedVert); glDeleteShader(basicTexturedFrag);
}

RenderState::RenderState(RenderContext* ctx_)
  : ctx(ctx_)
{
  for (int i=0; i<16; i++) mvp[i] = i/4 == i%4;
}

RenderState::RenderState(const RenderState& rs) {
  memcpy(mvp, rs.mvp, sizeof(rs.mvp));
  ctx = rs.ctx;
}
