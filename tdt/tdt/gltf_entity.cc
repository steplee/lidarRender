#include "tdt/gltf_entity.h"
#include "tdt/render_context.h"
#include <iostream>

#include "GL/glew.h"

#include "math.h"


#define BUFFER_OFFSET(i) ((char *)NULL + (i))

GltfEntity::GltfEntity() {
}

GltfEntity::~GltfEntity() {
  destroy();
}
void GltfEntity::destroy() {
  if (nvbos > 0) glDeleteBuffers(nvbos, vbos);
  if (ntexs > 0) glDeleteTextures(nvbos, vbos);
  nvbos = ntexs = 0;
}

void GltfEntity::upload(const GltfModel& model) {
  scenes = model.scenes;
  nodes = model.nodes;
  meshes = model.meshes;
  bufferViews = model.bufferViews;
  accessors = model.accessors;
  materials = model.materials;
  //prims = model.primitives;

  nvbos = model.buffers.size();
  ntexs = model.textures.size();
  vbos = (uint32_t*)malloc(sizeof(uint32_t)*nvbos);
  texs = (uint32_t*)malloc(sizeof(uint32_t)*nvbos);
  glGenBuffers(nvbos, vbos);
  glGenTextures(ntexs, texs);

  for (int i=0; i<nvbos; i++) {
    glBindBuffer(GL_ARRAY_BUFFER, vbos[i]);
    glBufferData(GL_ARRAY_BUFFER, model.buffers[i].byteLength, model.buffers[i].data.data(), GL_STATIC_DRAW);
    float* data = (float*)model.buffers[i].data.data();
    for (int i=0; i<10; i++) {
      std::cout << " - data[" << i << "] : " << data[i*3+0] << " " << data[i*3+1] << " " << data[i*3+2] << "\n";
    }
    CheckGLErrors("post bufferData");
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  for (int i=0; i<ntexs; i++) {
    glBindTexture(GL_TEXTURE_2D, texs[i]);
    int ifmt, fmt;
    const GltfTexture& t = model.textures[i];
    const GltfImage& im = model.images[i];
    ifmt = GL_RGB;
    if (im.channels == 1) fmt = GL_LUMINANCE;
    else if (im.channels == 3) fmt = GL_RGB;
    else if (im.channels == 4) fmt = GL_RGBA;
    else fmt = GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, im.w, im.h, 0, fmt, GL_UNSIGNED_BYTE, im.decodedData.data());
    std::cout << " - texImg " << im.w << " " << im.h << " " << ifmt << " " << fmt << "\n";
    bool needMipmap = false;
    if (t.sampler == -1) {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
      needMipmap |= (model.samplers[t.sampler].magFilter >= 0x2700 and model.samplers[t.sampler].magFilter <= 0x2703);
      needMipmap |= (model.samplers[t.sampler].minFilter >= 0x2700 and model.samplers[t.sampler].minFilter <= 0x2703);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, model.samplers[t.sampler].magFilter);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, model.samplers[t.sampler].minFilter);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, model.samplers[t.sampler].wrapS);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, model.samplers[t.sampler].wrapT);
    }
    if (needMipmap) glGenerateMipmap(GL_TEXTURE_2D);
    CheckGLErrors("post texImage");
    glBindTexture(GL_TEXTURE_2D, 0);
  }

}

void GltfEntity::renderAllNodes(const RenderState& rs) {
  for (int i=0; i<nodes.size(); i++) {
    renderNode(nodes[i], rs);
  }
}
void GltfEntity::renderScene(const RenderState& rs, int si) {
  auto& sceneNodes = scenes[si].baseNodes;
  for (int i=0; i<sceneNodes.size(); i++) {
    renderNode(nodes[sceneNodes[i]], rs);
  }
}


void GltfEntity::renderNode(const GltfNode& node, const RenderState& rs0) {
  RenderState rs(rs0);
  matmul44(rs.modelView, rs0.modelView, node.xform);

  //std::cout << " - renderNode.\n";

  if (node.mesh >= 0) {
    //const Shader& shader = rs0.ctx->basicShader;

    const GltfMesh& mesh = meshes[node.mesh];
    //std::cout << " - rendering mesh " << node.mesh << " : " << mesh.name <<  ", with " << mesh.primitives.size() << " prims" << ".\n";

    float mvp[16];
    matmul44_double_to_float(mvp, rs.proj, rs.modelView);
    //printf(" - MVP\n"); for (int i=0; i<16; i++) printf(" %f%s",mvp[i],i%4==3?"\n":"");

    for (int i=0; i<mesh.primitives.size(); i++) {
      //std::cout << " - prim " << i << "\n";
      const auto& prim = mesh.primitives[i];

      const Shader* shader = nullptr;

      bool haveColorAttrib = false;
      bool haveNormalAttrib = false;

      for (auto &ai : prim.attribs) {
        if (ai.attrib == AttributeType::COLOR_0) haveColorAttrib = true;
        if (ai.attrib == AttributeType::NORMAL) haveNormalAttrib = true;
      }

      if (prim.material == -1 and haveColorAttrib) {
        shader = &rs0.ctx->basicColorShader;
        glUseProgram(shader->id);
      }
      else if (prim.material == -1) {
        shader = &rs0.ctx->basicWhiteShader;
        glUseProgram(shader->id);
      } else {

        if (materials[prim.material].pbr.baseColorTexture.index == -1) {
          // If not diffuse texture, use colored shader.
          // NOTE: this is not exactly fool-proof.

          if (haveNormalAttrib) {
            shader = &rs0.ctx->litUniformColored;
            glUseProgram(shader->id);
          } else {
            shader = &rs0.ctx->basicUniformColorShader;
            glUseProgram(shader->id);
          }

          if (shader->u_color > -1) {
            //printf( " - u_color : %f %f %f %f\n", materials[prim.material].pbr.baseColorFactor[0], materials[prim.material].pbr.baseColorFactor[1], materials[prim.material].pbr.baseColorFactor[2], materials[prim.material].pbr.baseColorFactor[3]);
            glUniform4fv(shader->u_color, 1, materials[prim.material].pbr.baseColorFactor);
          }

        } else {
          shader = &rs0.ctx->defaultGltfShader;
          glUseProgram(shader->id);
          GLuint diffuse = texs[materials[prim.material].pbr.baseColorTexture.index];
          glBindTexture(GL_TEXTURE_2D, diffuse);
          glEnable(GL_TEXTURE_2D);
        }


      }

      if (shader->u_mvp > -1)
        glUniformMatrix4fv(shader->u_mvp, 1, true, mvp);
      if (shader->u_invModelViewT > -1) {
        float imvt[16];
        mat44_double_to_float_invt(imvt, rs.modelView);
        glUniformMatrix4fv(shader->u_invModelViewT, 1, true, imvt);
      }
      if (shader->u_modelView > -1) {
        float mv[16];
        for (int i=0; i<16; i++) mv[i] = rs.modelView[i];
        glUniformMatrix4fv(shader->u_modelView, 1, true, mv);
      }
      if (shader->u_view > -1) {
        glUniformMatrix4fv(shader->u_view, 1, true, rs.viewf);
      }
      if (shader->u_sunPos > -1) {
        float sunPos[3] = { .25, .25, .875 };
        glUniform3fv(shader->u_sunPos, 1, sunPos);
      }

      int count = 0; // If no indices provided, use count of position attr
      CheckGLErrors("pre set");

      for (const AttributeIndexPair& ai : prim.attribs) {
        const auto& acc = accessors[ai.id];
        const auto& bv = bufferViews[acc.bufferView];
        if (ai.attrib == AttributeType::POSITION) {
          glEnableVertexAttribArray(shader->in_pos);
          CheckGLErrors("enable in_pos");
          glBindBuffer(GL_ARRAY_BUFFER, vbos[bv.buffer]);
          glVertexAttribPointer(shader->in_pos, 3, GL_FLOAT, acc.normalized, bv.byteStride, BUFFER_OFFSET(acc.byteOffset+bv.byteOffset));
          CheckGLErrors("set attribPointer in_pos");
          count = acc.count;
        }
        if (ai.attrib == AttributeType::TEXCOORD_0) {
          if (shader->in_texcoord0 < 0) continue;
          glEnableVertexAttribArray(shader->in_texcoord0);
          glBindBuffer(GL_ARRAY_BUFFER, vbos[bv.buffer]);
          glVertexAttribPointer(shader->in_texcoord0, 2, GL_FLOAT, acc.normalized, bv.byteStride, BUFFER_OFFSET(acc.byteOffset+bv.byteOffset));
          CheckGLErrors("set in_texcoord0");
        }
        if (ai.attrib == AttributeType::TEXCOORD_1) {
          if (shader->in_texcoord1 < 0) continue;
          glEnableVertexAttribArray(shader->in_texcoord1);
          glBindBuffer(GL_ARRAY_BUFFER, vbos[bv.buffer]);
          glVertexAttribPointer(shader->in_texcoord1, 2, GL_FLOAT, acc.normalized, bv.byteStride, BUFFER_OFFSET(acc.byteOffset+bv.byteOffset));
          CheckGLErrors("set in_texcoord1");
        }
        if (ai.attrib == AttributeType::NORMAL) {
          if (shader->in_normal < 0) continue;
          glEnableVertexAttribArray(shader->in_normal);
          glBindBuffer(GL_ARRAY_BUFFER, vbos[bv.buffer]);
          glVertexAttribPointer(shader->in_normal, 3, GL_FLOAT, acc.normalized, bv.byteStride, BUFFER_OFFSET(acc.byteOffset+bv.byteOffset));
          CheckGLErrors("set in_normal");
        }
        if (ai.attrib == AttributeType::COLOR_0) {
          if (shader->in_color < 0) continue;
          glEnableVertexAttribArray(shader->in_color);
          glBindBuffer(GL_ARRAY_BUFFER, vbos[bv.buffer]);
          int n = acc.dataType == DataType::VEC4 ? 4 :
                  acc.dataType == DataType::VEC3 ? 3 : 1;
          glVertexAttribPointer(shader->in_color, n, GL_FLOAT, acc.normalized, bv.byteStride, BUFFER_OFFSET(acc.byteOffset+bv.byteOffset));
          CheckGLErrors("set in_color");
        }
      }

      if (prim.indices == -1) {
        glDrawArrays(prim.mode, 0, count);
        //std::cout << " - glDrawArrays(" << count << ").\n";
        CheckGLErrors("post drawArrays");
      } else {
        const GltfAccessor& indAcc = accessors[prim.indices];
        const GltfBufferView& indBv = bufferViews[indAcc.bufferView];
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbos[indBv.buffer]);
        glDrawElements(prim.mode, indAcc.count, indAcc.componentType, BUFFER_OFFSET(indAcc.byteOffset+indBv.byteOffset));
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        //std::cout << " - glDrawElements(" << indAcc.count << ").\n";
        CheckGLErrors("post drawElements");
      }


      for (const AttributeIndexPair& ai : prim.attribs) {
        if (ai.attrib == AttributeType::POSITION) glDisableVertexAttribArray(shader->in_pos);
      }
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glDisable(GL_TEXTURE_2D);

      //glVertexAttribPointer
    }

    glUseProgram(0);
  }
}
