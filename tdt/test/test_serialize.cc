#include "tdt/gltf.h"
#include <fstream>
#include <GL/glew.h>
#include "tdt/io.hpp"

int main() {

  Bytes bufBytes;

  GltfModel model(".");

  float verts[] = {
    -1,-1,0,  1,0,0,1,
     1,-1,0,  0,1,0,1,
     1, 1,0,  0,0,1,1,
     -1,1,0,  1,0,1,1
  };

  uint8_t inds[] = {
    0,1,2, 2,3,0
  };

  GltfBufferView bv, bv2;

  bufBytes.resize(4*4*7);
  memcpy(bufBytes.data(), verts, 4*4*7);
  bv.buffer = 0;
  bv.byteLength = sizeof(float) * 4 * 7;
  bv.byteOffset = 0;
  bv.byteStride = 4*7;
  bv.target = GL_ARRAY_BUFFER;

  bv2.buffer = 0;
  bv2.byteLength = 6;
  bv2.byteOffset = bufBytes.size();
  bv2.byteStride = 0;
  bv2.target = GL_ELEMENT_ARRAY_BUFFER;
  bufBytes.resize(bufBytes.size() + 6);
  memcpy(bufBytes.data()+bufBytes.size() - 6, inds, 6);
  while (bufBytes.size() % 4 != 0) bufBytes.push_back(0);


  GltfAccessor acc_pos, acc_color, acc_ind;

  acc_pos.bufferView = 0;
  acc_pos.byteOffset = 0;
  acc_pos.attributeType = AttributeType::POSITION;
  acc_pos.dataType = DataType::VEC3;
  acc_pos.componentType = GL_FLOAT;
  acc_pos.count = 4;
  acc_pos.normalized = false;

  acc_color.bufferView = 0;
  acc_color.byteOffset = 4*3;
  acc_color.attributeType = AttributeType::COLOR_0;
  acc_color.dataType = DataType::VEC4;
  acc_color.componentType = GL_FLOAT;
  acc_color.count = 4;
  acc_color.normalized = false;

  acc_ind.bufferView = 1;
  acc_ind.byteOffset = 0;
  acc_ind.dataType = DataType::SCALAR;
  acc_ind.componentType = GL_UNSIGNED_BYTE;
  acc_ind.count = 6;
  acc_ind.normalized = false;

  model.bufferViews.push_back(bv);
  model.bufferViews.push_back(bv2);
  model.accessors.push_back(acc_pos);
  model.accessors.push_back(acc_color);
  model.accessors.push_back(acc_ind);

  double eye[] = {
    1,0,0,0,
    0,1,0,0,
    0,0,1,0,
    0,0,0,1};
  double trans_z_COLMAJOR[] = {
    .3,0,0,0,
    0,.3,0,0,
    0,0,.3,0,
    0,0,1,1};

  GltfPrimitive prim;
  prim.indices = 2;
  prim.material = -1;
  prim.mode = GL_TRIANGLES;
  prim.attribs.push_back(AttributeIndexPair{AttributeType::POSITION, 0});
  prim.attribs.push_back(AttributeIndexPair{AttributeType::COLOR_0, 1});
  std::vector<GltfPrimitive> prims { prim };
  GltfMesh mesh { prims, "mesh0" };
  model.meshes.push_back(mesh);

  GltfNode node1;
  node1.mesh = 0;
  memcpy(node1.xform, trans_z_COLMAJOR, sizeof(double)*16);
  model.nodes.push_back(node1);


  // Create second mesh, which is textured
  {
    float verts_uv[] = {
    -1,-1,0,  0,1,
     1,-1,0,  1,1,
     1, 1,0,  1,0,
     -1,1,0,  0,0 };
    GltfBufferView bv3, bv4;
    bv3.buffer = 0;
    bv3.byteLength = sizeof(float) * 4 * 5;
    bv3.byteOffset = bufBytes.size();
    bv3.byteStride = 4*5;
    bv3.target = GL_ARRAY_BUFFER;
    model.bufferViews.push_back(bv3);
    bufBytes.resize(bufBytes.size() + sizeof(verts_uv));
    memcpy(bufBytes.data() + bufBytes.size() - sizeof(verts_uv), verts_uv, sizeof(verts_uv));
    while (bufBytes.size() % 4 != 0) bufBytes.push_back(0);

    Bytes texData;
    readFile(texData, "/home/slee/Pictures/me2.jpg");
    bv4.buffer = 0;
    bv4.byteLength = texData.size();
    bv4.byteOffset = bufBytes.size();
    bv4.byteStride = 0;
    model.bufferViews.push_back(bv4);
    //bv4.byteStride = 4*5;
    //bv4.target = GL_ARRAY_BUFFER;
    bufBytes.resize(bufBytes.size() + texData.size());
    memcpy(bufBytes.data() + bufBytes.size() - texData.size(), texData.data(), texData.size());
    while (bufBytes.size() % 4 != 0) bufBytes.push_back(0);


    GltfAccessor acc_pos3, acc_uv3;
    acc_pos3.bufferView = 2;
    acc_pos3.byteOffset = 0;
    acc_pos3.attributeType = AttributeType::POSITION;
    acc_pos3.dataType = DataType::VEC3;
    acc_pos3.componentType = GL_FLOAT;
    acc_pos3.count = 4;
    acc_pos3.normalized = false;

    acc_uv3.bufferView = 2;
    acc_uv3.byteOffset = 4*3;
    acc_uv3.attributeType = AttributeType::TEXCOORD_0;
    acc_uv3.dataType = DataType::VEC2;
    acc_uv3.componentType = GL_FLOAT;
    acc_uv3.count = 4;
    acc_uv3.normalized = false;

    model.accessors.push_back(acc_pos3);
    model.accessors.push_back(acc_uv3);

    GltfTexture tex;
    tex.sampler = -1;
    tex.source = 0;
    GltfImage image;
    image.bufferView = 3;
    image.mimeType = "image/jpeg";
    model.textures.push_back(tex);
    model.images.push_back(image);

    GltfMaterial mat;
    mat.name = "textured1";
    GltfTextureRef texRef;
    texRef.index = 0;
    mat.pbr.baseColorTexture = texRef;
    model.materials.push_back(mat);

    // Re-use indices
    GltfPrimitive prim;
    prim.indices = 2;
    prim.material = 0;
    prim.mode = GL_TRIANGLES;
    prim.attribs.push_back(AttributeIndexPair{AttributeType::POSITION, 3});
    prim.attribs.push_back(AttributeIndexPair{AttributeType::TEXCOORD_0, 4});
    std::vector<GltfPrimitive> prims { prim };
    GltfMesh mesh2 { prims, "mesh2" };
    model.meshes.push_back(mesh2);

    GltfNode node2;
    node2.mesh = 1;
    memcpy(node2.xform, eye, sizeof(double)*16);
    model.nodes.push_back(node2);
  }

  model.scenes.push_back(GltfScene{"scene1", {0,1}});

  while (bufBytes.size() % 4 != 0) bufBytes.push_back(0);
  model.buffers.push_back(GltfBuffer{bufBytes, bufBytes.size()});

  std::string s = model.serialize_glb();
  std::ofstream of("tst.glb"); of << s;




  return 0;
}

