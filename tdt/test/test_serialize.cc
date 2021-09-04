#include "tdt/gltf.h"
#include <fstream>
#include <GL/glew.h>

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
  bv.buffer = 0;
  bv.byteLength = sizeof(float) * 4 * 7;
  bv.byteOffset = 0;
  bv.byteStride = 4*7;
  bv.target = GL_ARRAY_BUFFER;
  bv2.buffer = 0;
  bv2.byteLength = 6;
  bv2.byteOffset = 4*4*7;
  bv2.byteStride = 0;
  bv2.target = GL_ELEMENT_ARRAY_BUFFER;


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

  GltfNode node2;
  node2.mesh = 0;
  memcpy(node2.xform, eye, sizeof(double)*16);
  model.nodes.push_back(node2);
  model.scenes.push_back(GltfScene{"scene1", {0,1}});

  bufBytes.resize(4*4*7 + 6);
  memcpy(bufBytes.data(), verts, 4*4*7);
  memcpy(bufBytes.data()+4*4*7, inds, 6);
  model.buffers.push_back(GltfBuffer{bufBytes, bufBytes.size()});

  std::string s = model.serialize_glb();
  std::ofstream of("tst.glb"); of << s;




  return 0;
}

