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
    std::cout << " - Error linking shader " << name << ":\n";
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

void Shader::compile(const std::string& vs, const std::string& fs) {
  GLuint vid, fid;
  LoadShaderFromString(name + "_vs", GL_VERTEX_SHADER, vid, vs);
  LoadShaderFromString(name + "_fs", GL_FRAGMENT_SHADER, fid, fs);
  compile(vid, fid);
  glDeleteShader(vid);
  glDeleteShader(fid);
}

#define GET_UNIFORM(a) a = glGetUniformLocation(id, #a );
#define GET_ATTRIB(a) a = glGetAttribLocation(id, #a );
void Shader::findBindings() {
  CheckGLErrors("pre-find bindings.");

  GET_UNIFORM(u_mvp);
  GET_UNIFORM(u_modelView);
  GET_UNIFORM(u_invModelViewT);
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

const std::string basicColorVertSrc = R"(
#version 440
in vec3 in_pos;
in vec4 in_color;
uniform mat4 u_mvp;
out vec4 v_color;
void main() {
  gl_Position = u_mvp * vec4(in_pos,1.0);
  v_color = in_color;
}
)";
const std::string basicColorFragSrc = R"(
#version 440
in vec4 v_color;
out vec4 outColor;
void main() {
  outColor = v_color;
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

//#define HIGH_QUALITY

#ifdef HIGH_QUALITY
const std::string litUniformColored_VS = R"(
#version 440
in vec3 in_pos;
in vec3 in_normal;
uniform mat4 u_mvp;
out vec3 v_normal;

void main() {
  gl_Position = u_mvp * vec4(in_pos,1.0);
  v_normal = in_normal;

}
)";
const std::string litUniformColored_FS = R"(
#version 440
in vec3 v_normal;

uniform mat4 u_view;
uniform mat4 u_invModelViewT;
uniform vec3 u_sunPos;
uniform vec4 u_color;

void main() {

  vec3 nrml = normalize(mat3(u_invModelViewT) * v_normal);

  vec3 amb = .2 * u_color.rgb;
  vec3 dif = .6 * u_color.rgb * clamp(dot(nrml, -u_sunPos), 0.,1.);

  vec3 zplus = vec3(u_view[2].x, u_view[2].y, u_view[2].z);
  //vec3 zplus = vec3(u_view[2].xyz);
  vec3 specDir = normalize(-u_sunPos + zplus);
  vec3 spec = .72 * vec3(1.,1.,.8) * pow(clamp(dot(nrml, specDir), 0.,1.), 15.);
  vec4 color = clamp(vec4(dif+spec+amb, u_color.a), 0., 1.);

  gl_FragColor = color;
  }
)";
#else
// Does lighting calculations in vertex shader, which is lwoer quality but easier to see LoD
const std::string litUniformColored_VS = R"(
#version 440
in vec3 in_pos;
in vec3 in_normal;
uniform mat4 u_mvp;
uniform mat4 u_view;
uniform mat4 u_invModelViewT;
uniform vec3 u_sunPos;
uniform vec4 u_color;
out vec4 v_color;

void main() {
  gl_Position = u_mvp * vec4(in_pos,1.0);

  vec3 nrml = normalize(mat3(u_invModelViewT) * in_normal);
  //vec3 nrml = in_normal;

  vec3 amb = .2 * u_color.rgb;
  vec3 dif = .6 * u_color.rgb * clamp(dot(nrml, -u_sunPos), 0.,1.);

  vec3 specDir = normalize(-u_sunPos + (u_view)[2].xyz);
  vec3 spec = .92 * vec3(1.,1.,.8) * pow(clamp(dot(nrml, specDir), 0.,1.), 15.);
  v_color = clamp(vec4(dif+spec+amb, v_color.a), 0., 1.);
}
)";
const std::string litUniformColored_FS = R"(
#version 440
in vec4 v_color;
void main() {
  gl_FragColor = v_color;
  }
)";
#endif

void RenderContext::compileShaders() {


  CheckGLErrors("pre compile");

  GLuint basicVertId = 0;
  GLuint basicWhiteFragId = 0;
  GLuint basicUniformColorFragId = 0;
  LoadShaderFromString("basicVert", GL_VERTEX_SHADER, basicVertId, basicVertSrc);
  LoadShaderFromString("basicWhiteFrag", GL_FRAGMENT_SHADER, basicWhiteFragId, basicFragSrc);
  LoadShaderFromString("basicUniformColorFrag", GL_FRAGMENT_SHADER, basicUniformColorFragId, basicUniformColorFragSrc);
  basicWhiteShader.compile(basicVertId, basicWhiteFragId);
  basicUniformColorShader.compile(basicVertId, basicUniformColorFragId);

  GLuint basicColorVertId = 0;
  GLuint basicColorFragId = 0;
  LoadShaderFromString("basicColorFrag", GL_FRAGMENT_SHADER, basicColorFragId, basicColorFragSrc);
  LoadShaderFromString("basicColorVert", GL_VERTEX_SHADER, basicColorVertId, basicColorVertSrc);
  basicColorShader.compile(basicColorVertId, basicColorFragId);

  GLuint basicTexturedVert = 0;
  GLuint basicTexturedFrag = 0;
  LoadShaderFromString("basicTexturedVert", GL_VERTEX_SHADER, basicTexturedVert, basicTexturedVertSrc);
  LoadShaderFromString("basicTexturedWhiteFrag", GL_FRAGMENT_SHADER, basicTexturedFrag, basicTexturedFragSrc);

  defaultGltfShader.compile(basicTexturedVert, basicTexturedFrag);
  basicTexturedShader.compile(basicTexturedVert, basicTexturedFrag);

  litUniformColored.compile(litUniformColored_VS, litUniformColored_FS);

  glDeleteShader(basicVertId); glDeleteShader(basicWhiteFragId);
  glDeleteShader(basicColorVertId); glDeleteShader(basicColorFragId);
  glDeleteShader(basicUniformColorFragId);
  glDeleteShader(basicTexturedVert); glDeleteShader(basicTexturedFrag);
}

RenderState::RenderState(RenderContext* ctx_)
  : ctx(ctx_)
{
  for (int i=0; i<16; i++) modelView[i] = i/4 == i%4;
  for (int i=0; i<16; i++) proj[i] = i/4 == i%4;
}

RenderState::RenderState(const RenderState& rs) {
  //memcpy(mvp, rs.mvp, sizeof(rs.mvp));
  memcpy(modelView, rs.modelView, sizeof(rs.modelView));
  memcpy(proj, rs.proj, sizeof(rs.proj));
  memcpy(viewf, rs.viewf, sizeof(rs.viewf));
  ctx = rs.ctx;
}



void RenderEngine::make(int w, int h) {
  glGenFramebuffers(1, &fbo);
  glGenTextures(1, &frameRgbTex);
  glGenTextures(1, &frameDepthTex);
  glBindTexture(GL_TEXTURE_2D, frameRgbTex);
  CheckGLErrors("post bind frgbtex")
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8, w, h);
  CheckGLErrors("post texStorage rgb8")
  glBindTexture(GL_TEXTURE_2D, frameDepthTex);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32, w, h);
  CheckGLErrors("post texStorage depth32")
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, frameRgbTex, 0);
  CheckGLErrors("post fbo set tex color")
  glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  frameDepthTex, 0);
  CheckGLErrors("post fbo set tex depth")
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  CheckGLErrors("post make fbo")

  /* glBindTexture(GL_TEXTURE_2D, frameRgbTex);
  uint8_t* d = (uint8_t*)malloc(w*h*3);
  for (int i=0; i<w*h*3; i++) d[i] = (i*77) % 255;
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, w,h, GL_RGB, GL_UNSIGNED_BYTE, d);
  glBindTexture(GL_TEXTURE_2D, 0);
  free(d); */
}

void RenderEngine::setTarget() {
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  CheckGLErrors("re::setTarget");
}
void RenderEngine::unsetTarget() {
  CheckGLErrors("pre re::unsetTarget");
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  CheckGLErrors("post re::unsetTarget");
}
void RenderEngine::renderToScreen(RenderContext& rs) {
  CheckGLErrors("pre re::renderToScreen")
  // Render full-screen quad with the rgb tex.
  static constexpr float quad[] = {
    -1, -1, 0, 0,0,
     1, -1, 0, 1,0,
     1,  1, 0, 1,1,
    -1,  1, 0, 0,1,
  };
  static constexpr float mvp[] = {
    1,0,0,0,
    0,1,0,0,
    0,0,1,0,
    0,0,0,1,
  };
  static constexpr uint8_t inds[] = {
    0,1,2, 2,3,0
  };

  glClearColor(0,0,0,1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  auto &shader = rs.basicTexturedShader;
  glUseProgram(shader.id);
  glEnableVertexAttribArray(shader.in_pos);
  glEnableVertexAttribArray(shader.in_texcoord0);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, frameRgbTex);
  glUniformMatrix4fv(shader.u_mvp, 1, true, mvp);
  glVertexAttribPointer(shader.in_pos, 3, GL_FLOAT, false, 4*5, quad);
  glVertexAttribPointer(shader.in_texcoord0, 2, GL_FLOAT, false, 4*5, quad+3);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, inds);
  glDisableVertexAttribArray(shader.in_pos);
  glDisableVertexAttribArray(shader.in_texcoord0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);
  glUseProgram(0);
  CheckGLErrors("post re::renderToScreen")
}
