#include "tdt/gltf.h"
#include <fstream>

int main() {

  Bytes bufBytes;
  bufBytes.push_back(0x01);
  bufBytes.push_back(0x02);
  bufBytes.push_back(0x3);
  bufBytes.push_back(0x4);
  bufBytes.push_back(0x5);
  bufBytes.push_back(0x6);
  bufBytes.push_back(0x7);
  bufBytes.push_back(0x8);

  GltfModel model(".");
  model.scenes.push_back(GltfScene{"scene1", {0}});
  model.buffers.push_back(GltfBuffer{bufBytes, bufBytes.size()});

  std::string s = model.serialize_glb();
  std::ofstream of("tst.glb"); of << s;




  return 0;
}

